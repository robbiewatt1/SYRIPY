from ERS import Wavefront, FieldSolver
from ERS.Optics import FraunhoferProp, CircularAperture, OpticsContainer
from ERS.Tracking import Track, Dipole, FieldContainer
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
                         torch.tensor([0, 0.49051235, 0]), None, 0.05, device)
        dipole1 = Dipole(torch.tensor([0, 0, 1.0334]), 0.203274830142196,
                         torch.tensor([0, -0.49051235, 0]), None, 0.05, device)
        dipole2 = Dipole(torch.tensor([0, 0, 2.0668]), 0.203274830142196,
                         torch.tensor([0, -0.49051235, 0]), None, 0.05, device)
        self.field = FieldContainer([dipole0, dipole1, dipole2])

        # Define a simple optics beamline with an aperture and a
        # propagation to focus
        self.optics = OpticsContainer([CircularAperture(0.02),
                                       FraunhoferProp(1)])

        # Set the known / initial assumption beam parameters for the simulation
        self.time = torch.linspace(0, 18, 1001)
        self.d0 = torch.tensor([-0, 0, 1.])
        self.r0 = torch.tensor([-0.1155863873, 0, -1.00])
        self.g0 = 339.3 / 0.51099890221

        # Define the track / wavefront  / solver classes for the simulation
        self.track = Track(device=device)
        self.wavefront = Wavefront(3.786062584928902, 3.77e5,
                                   [-0.02, 0.02, -0.02, 0.02],
                                   [300, 1], device=device)
        self.solver = FieldSolver(self.wavefront, self.track)

        # Finally we wrap the solver function with torch.func.vmap, allowing the
        # calculation to be batched.
        def solver_func(index):
            wvfrt = self.solver.solve_field(1, True, True, index)
            self.optics.propagate(wvfrt)
            return wvfrt.get_intensity()
        self.solver_func = torch.func.vmap(solver_func)

    def forward(self, dir_x, div_x):
        """
        This represents a forward call to the model. In this tutorial we are
        interested in inferring the direction and divergence of the beam in the
        x-direction, so these are the two input parameters of this function.
        :param dir_x: X direction of beam, i.e. the mean x-direction of
         particles
        :param div_x: X divergence of the beam, i.e. the standard deviation of
         the x-direction of the particles
        :return:
        """

        # We start by generating the initial conditions for the beam. This
        # involves sampling from a standard normal and shifting / rescaling by
        # dir_x and div_x respectively
        bunch_d = self.d0 + div_x * torch.tensor([1., 0., 0.]) * torch.randn(
            self.samples)[:, None] + torch.tensor([1., 0., 0.]) * dir_x
        bunch_r = self.r0.repeat(self.samples, 1)
        bunch_g = torch.ones(self.samples) * self.g0

        # Now we simulate the tracks. Can't use the c++  version as gradients
        # are required
        self.track.sim_bunch(self.field, bunch_r, bunch_d, bunch_g, self.time)

        # Now we down-sample the track so the field solver is faster
        self.solver.set_dt(201, t_start=4, t_end=13, set_bunch=True)

        # Finally we calculate the radiation intensity for the beam
        intensity_total = torch.zeros((300, 1), device=self.track.device)
        for batch in range(0, int(self.samples / self.n_batch)):
            batch_idx = torch.arange(batch * self.n_batch,
                                     (batch + 1) * self.n_batch)
            intensity_total += torch.mean(self.solver_func(batch_idx), dim=0)
        return intensity_total.T / self.n_batch


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
    dir = pyro.sample('Dir', dist.Uniform(-3e-3, 3e-3))
    div = pyro.sample('Div', dist.Uniform(0, 1e-3))
    sigma = pyro.sample("Sigma", dist.Uniform(0.00, 0.3))

    # Now call the forward model wiht the sample
    predict = forward_model.forward(dir, div).to("cpu")
    predict = predict / torch.max(predict)

    # Now sample the observed random variable
    with pyro.plate("data", predict.shape[1]):
        return pyro.sample('obs', dist.Normal(predict, sigma),
                           obs=measured_intensity)


if __name__ == "__main__":
    # First we will start by defining the forward model and generating some mock
    # experimental data.

    # 500 macro-electrons seems to be enough and all fits on my GPU
    forward_model = ForwardModel(500, 500)

    # Model a beam that is 1.2 mrad off axis with 300urad divergence and
    # normalise
    intensity = forward_model.forward(1.2e-3, 0.30e-3)
    intensity = intensity / torch.max(intensity)

    # Add some noise to the data
    intensity = torch.normal(intensity, 0.05).to("cpu")

    # Check what the mock data looks like
    fig, ax = plt.subplots()
    ax.plot(intensity.cpu()[0], color="red")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("I [a.u.]")
    ax.set_xlim([0, 300])
    plt.show()

    # We are going to be using variational inference. Therefore, we need to
    # define the variational distribution (guide), optimiser (adam) and loss
    # function (ELBO which is a lower bound of the KL divergence).
    auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(pryo_model)
    adam = pyro.optim.Adam({"lr": 0.05})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(pryo_model, auto_guide, adam, elbo)

    # We now run the optimiser for some number of steps
    losses = []
    for step in range(50):
        loss = svi.step(forward_model, intensity)
        print(step, loss)
        losses.append(loss)

    # We can now sample from the guide distribution (i.e. approximation to the
    # posterior distribution)
    with pyro.plate("samples", 1000, dim=-1):
        samples = auto_guide(forward_model)

    # Convert the samples to pandas for plotting
    samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}
    data = pandas.DataFrame(samples)

    # Plot the posterior marginals
    g = seaborn.pairplot(data, corner=True, diag_kind="kde")
    g.map_upper(seaborn.scatterplot, s=15)
    g.map_lower(seaborn.kdeplot)
    g.map_diag(seaborn.kdeplot, lw=2)

    # Run the forward model with the mean of the posterior and compair it to the
    # mock data
    dir_mean = np.mean(samples["Dir"])
    div_mean = np.mean(samples["Div"])
    intensity2 = forward_model.forward(dir_mean, div_mean)
    intensity2 = intensity2 / torch.max(intensity2)
    fig, ax = plt.subplots()
    ax.plot(intensity.cpu()[0], color="red", label="Measured")
    ax.plot(intensity2.cpu()[0], color="black", label="Predicted")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("I [a.u.]")
    ax.set_xlim([0, 300])
    ax.legend()
    plt.show()
