from SYRIPY import Wavefront, FieldSolver
from SYRIPY.Optics import FraunhoferProp, CircularAperture, OpticsContainer
from SYRIPY.Tracking import Track, Dipole, FieldContainer
import torch
import numpy as np
import pyro
import pyro.distributions as dist
import seaborn
import pandas
import matplotlib.pyplot as plt

# In this example we will show how the package can be used to perform
# Bayesian inference on mock experimental data. For this we will use the
# probabilistic program language pyro, which uses stochastic variational
# inference. Pyro, Seaborn and Pandas are not part of the package requirements
# and need to be installed separately.

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class ForwardModel:
    """
    This class builds the forward model of the system. This means it takes in
    inputs (which we want to infer using Bayesian inference) and returns the
    expected signal. For this example we will use a setup with two dipoles and
    look at the interference edge pattern. We will restrict  the problem to 1D
    so the calculation can be performed in a reasonable time.
    """

    def __init__(self, samples, n_batch):
        """
        :param samples: Number of macro-electron samples from the beam per
         simulation
        :param n_batch: Number of macro-electrons simulated per batch. (This
         allows multiple macro-electrons to be simulated at once but requires
         more memory)
        """
        self.samples = samples
        self.n_batch = n_batch

        # First we set up the magnetic field
        dipole0 = Dipole(torch.tensor([0, 0, 0]), 0.203274830142196,
                         torch.tensor([0, 0.49051235, 0]), None, 0.05)
        dipole1 = Dipole(torch.tensor([0, 0, 1.0334]), 0.203274830142196,
                         torch.tensor([0, -0.49051235, 0]), None, 0.05)
        dipole2 = Dipole(torch.tensor([0, 0, 2.0668]), 0.203274830142196,
                         torch.tensor([0, -0.49051235, 0]), None, 0.05)
        self.field = FieldContainer([dipole0, dipole1, dipole2])

        # Set the know central beam parameters for the simulation
        self.time = torch.linspace(0, 18, 401)
        self.g0 = torch.tensor([339.3 / 0.51099890221])
        self.r0 = torch.tensor([-0.091386, 0, -1.00])
        self.d0 = torch.tensor([-0, 0, 1.])

        # Define the track / wavefront  / solver classes for the simulation
        self.track = Track(self.field, device=device)
        self.track = torch.jit.script(self.track)
        self.wavefront = Wavefront(3.086, 3.77e5,
                                   [-0.02, 0.02, -0.02, 0.02],
                                   [300, 1], device=device)
        self.solver = FieldSolver(self.wavefront, self.track)

        # Finally we wrap the solver function with torch.func.vmap, allowing the
        # calculation to be batched.
        def solver_func(index):
            wvfrt = self.solver.solve_field_vmap(index)
            return wvfrt.get_intensity()
        self.solver_func = torch.func.vmap(solver_func)

    def forward(self, size_x, div_x):
        """
        This represents a forward call to the model. In this tutorial we are
        interested in inferring the size and divergence of the beam in the
        x-direction, so these are the two input parameters of this function.
        :param size_x: X size of beam
        :param div_x: X divergence of the beam
        :return: predicted intensity
        """

        # We start by generating the initial conditions for the beam. This
        # involves sampling from a standard normal and shifting / rescaling by
        # size_x and div_x respectively
        bunch_d = self.d0 + div_x * torch.tensor([1., 0., 0.]) * torch.randn(
            self.samples)[:, None]
        bunch_p = bunch_d * self.g0 * 0.29979
        bunch_r = self.r0 + size_x * torch.tensor([1., 0., 0.]) * torch.randn(
            self.samples)[:, None]

        # Now we simulate the tracks. Can't use the c++  version as gradients
        # are required
        self.track.sim_beam(self.time, beam_r=bunch_r, beam_p=bunch_p)

        # Now we down-sample the track so the field solver is faster
        self.solver.set_track(201, t_start=4, t_end=13)

        # Finally we calculate the radiation intensity for the beam
        intensity_total = torch.zeros((300, 1), device=self.track.device)
        for batch in range(0, int(self.samples / self.n_batch)):
            batch_idx = torch.arange(batch * self.n_batch,
                                     (batch + 1) * self.n_batch)
            intensity_total += torch.mean(self.solver_func(batch_idx), dim=0)
        return intensity_total.T / int(self.samples / self.n_batch)


def pryo_model(forward_model, measured_intensity=None):
    """
    This is a probabilistic model defining our random variables and the
    dependencies between them (i.e. p(X, I) where X are the variables we want to
    infer and I is our observed Intensity)
    :param forward_model: Instance of the ForwardModel class defined above
    :param measured_intensity: Measured observable that we want to infer
     parameters from
    :return: Sample of observable variable
    """
    # Start by sampling from the latent variables with sensible uniform priors
    size = pyro.sample('Size', dist.Uniform(0, 1e-3))
    div = pyro.sample('Div', dist.Uniform(0, 1e-3))
    sigma = pyro.sample("Sigma", dist.Uniform(0.00, 0.003))

    # Now call the forward model with the sample
    predict = forward_model.forward(size, div).to("cpu")
    predict = predict / torch.trapz(predict)

    # Now sample the observed random variable
    with pyro.plate("data", predict.shape[1]):
        return pyro.sample('obs', dist.Normal(predict, sigma),
                           obs=measured_intensity)


if __name__ == "__main__":
    # First we will start by defining the forward model and generating some mock
    # experimental data.

    # 1000 macro-electrons seems to be enough and all fits on my GPU
    forward_model = ForwardModel(1000, 1000)

    # Model a beam that with 500um size and 150urad divergence
    intensity = forward_model.forward(0.5e-3, 0.15e-3)
    intensity = intensity / torch.trapz(intensity)

    # Add some noise to the data
    intensity = torch.normal(intensity, 0.0005).to("cpu")

    # Check what the mock data looks like
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(intensity.cpu()[0], color="red")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("I [a.u.]")
    ax.set_xlim([0, 300])
    fig.tight_layout()
    plt.show()

    # We are going to be using variational inference. Therefore, we need to
    # define the variational distribution (guide), optimiser (adam) and loss
    # function (ELBO which is a lower bound of the KL divergence).
    auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(pryo_model)
    adam = pyro.optim.Adam({"lr": 0.04})
    elbo = pyro.infer.Trace_ELBO(num_particles=4)
    svi = pyro.infer.SVI(pryo_model, auto_guide, adam, elbo)

    # We now run the optimiser for some number of steps
    losses = []
    for step in range(100):
        loss = svi.step(forward_model, intensity)
        print(step, loss)
        losses.append(loss)
    fig, ax = plt.subplots()
    ax.plot(losses)

    # We can now sample from the guide distribution (i.e. approximation to the
    # posterior distribution)
    with pyro.plate("samples", 10000, dim=-1):
        samples = auto_guide(forward_model)

    # Convert the samples to pandas for plotting
    samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}
    data = pandas.DataFrame(samples)
    data.to_csv("./samples.csv")

    # Plot the posterior marginals
    g = seaborn.pairplot(data, corner=True, diag_kind="kde")
    g.map_lower(seaborn.kdeplot)
    g.map_diag(seaborn.kdeplot, lw=2)

    # Run the forward model with the mean of the posterior and compair it to the
    # mock data
    dir_mean = np.mean(samples["Size"])
    div_mean = np.mean(samples["Div"])
    intensity2 = forward_model.forward(dir_mean, div_mean)
    intensity2 = intensity2 / torch.trapz(intensity2)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(intensity.cpu()[0], color="red", label="Measured")
    ax.plot(intensity2.cpu()[0], color="black", label="Predicted")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("I [a.u.]")
    ax.set_xlim([0, 300])
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()
