#include "Track.hh"
#include <fstream>
#include <stdexcept>
#include <chrono>


#ifdef _OPENMP
    #include <omp.h>
#endif

py::tuple Track::simulateTrack()
{
#ifdef USE_TORCH
    torch::AutoGradMode enable_grad(true);
#endif
    if (!m_timeSet || !m_initSet || !m_fieldSet)
    {
        throw std::runtime_error("Error: Time or initial conditions not set.");
    }

    m_position = std::vector<vectorType>(m_time.size());
    m_momentum = std::vector<vectorType>(m_time.size());
    m_beta = std::vector<vectorType>(m_time.size());

    m_position[0] = m_initPosition;
    m_momentum[0] = m_initMomemtum;
    m_beta[0] = m_initMomemtum / math::sqrt(c_light * c_light
        + m_initMomemtum.Magnitude2());
    scalarTensor gamma = math::sqrt(1. + m_initMomemtum.Magnitude2()
        / (c_light * c_light));

    for (long unsigned int i = 0; i < m_time.size() - 1; i++)
    {
        updateTrack(m_position, m_momentum, m_beta, gamma, i);
    }
    
    // Convert and return
    int samples = m_time.size();
    py::array_t<scalarType> positionReturn({samples, 3});
    py::array_t<scalarType> betaReturn({samples, 3});
    auto positionMutable = positionReturn.mutable_unchecked<2>();
    auto betaMutable = betaReturn.mutable_unchecked<2>();

    for (long unsigned int i = 0; i < m_time.size(); i++)
    {
        scalarType* pos_ptr = m_position[i].data_ptr<scalarType>();
        scalarType* beta_ptr = m_beta[i].data_ptr<scalarType>();
        positionMutable(i, 0) = pos_ptr[0];
        positionMutable(i, 1) = pos_ptr[1];
        positionMutable(i, 2) = pos_ptr[2];
        betaMutable(i, 0) = beta_ptr[0];
        betaMutable(i, 1) = beta_ptr[1];
        betaMutable(i, 2) = beta_ptr[2];
    }
    return py::make_tuple(m_time, positionReturn, betaReturn);
}

py::tuple Track::backwardTrack(py::array_t<scalarType> &positionGrad,
    py::array_t<scalarType> &betaGrad)
{
#ifndef USE_TORCH
    throw std::runtime_error("Gradient through track is not supported"
     " without installing libtorch version of SYRIPY.");
#else
    // check that forward track has been run
    if (m_position.size() == 0)
    {
        throw std::runtime_error("Error: Can't run backward through track"
            "without running forward first.");
    }
     // Convert to vector of Tensors
    std::vector<torch::Tensor> tensor_position(m_position.begin(),
        m_position.end());
    std::vector<torch::Tensor> tensor_momenutm(m_beta.begin(),
        m_beta.end());
    std::vector<torch::Tensor> tensor_cat = tensor_position;
    tensor_cat.insert(tensor_cat.end(), tensor_momenutm.begin(),
        tensor_momenutm.end());

    // Create the grad_outputs
    std::vector<torch::Tensor> grad_outputs;
    auto buffer_info = positionGrad.request();
    auto positionPtr = static_cast<scalarType*>(buffer_info.ptr);
    for (ssize_t i = 0; i <  buffer_info.shape[0]; ++i)
    {
        auto tensor = vectorType(&positionPtr[i*3]);
        grad_outputs.push_back(tensor);
    }
    buffer_info = betaGrad.request();
    auto momentumPtr = static_cast<scalarType*>(buffer_info.ptr);
    for (ssize_t i = 0; i < buffer_info.shape[0]; ++i)
    {
        auto tensor = vectorType(&momentumPtr[i*3]);
        grad_outputs.push_back(tensor);
    }

    std::vector<torch::Tensor> inputs = {m_initPosition, m_initMomemtum};

    pybind11::gil_scoped_release no_gil;
    auto gradients = torch::autograd::grad(tensor_cat, inputs, grad_outputs);
    pybind11::gil_scoped_acquire acquire_gil;

    // Need to check if both params require grads or just 1
    py::array_t<scalarType> position_grad = py::array_t<scalarType>(3,
        gradients[0].data_ptr<scalarType>());
    py::array_t<scalarType> momentum_grad = py::array_t<scalarType>(3,
        gradients[1].data_ptr<scalarType>());

    return py::make_tuple(position_grad, momentum_grad);
#endif
}

py::tuple Track::simulateBeam()
{
#ifdef USE_TORCH
    torch::AutoGradMode enable_grad(true);
#endif
    if (!m_timeSet || !m_beamSet || !m_fieldSet)
    {
        throw std::runtime_error("Error: Time or initial conditions not set.");
    }
    int nPart = m_initBeamPosition.size();
    int tSamp = m_time.size();
    m_beamPosition = std::vector<vectorType>(nPart * tSamp);
    m_beamMomentum = std::vector<vectorType>(nPart * tSamp);
    m_beamBeta = std::vector<vectorType>(nPart * tSamp);

#ifdef _OPENMP
    #pragma omp parallel for num_threads(32)
#endif
    for (int i = 0; i < nPart; i++)
    {
        int partIndex = i * tSamp;
        m_beamPosition[partIndex] = m_initBeamPosition[i];
        m_beamMomentum[partIndex] = m_initBeamMomentum[i];
        m_beamBeta[partIndex] = m_initBeamMomentum[i] / math::sqrt(
            c_light * c_light + m_initBeamMomentum[i].Magnitude2());
        scalarTensor gamma = math::sqrt(1. + m_initBeamMomentum[i].Magnitude2()
            / (c_light * c_light));
        for (long unsigned int j = 0; j < tSamp - 1; j++)
        {
            updateTrack(m_beamPosition, m_beamMomentum, m_beamBeta, gamma,
                partIndex + j);
        }
    }
    // Convert and return
    int samples = m_time.size();
    py::array_t<scalarType> positionReturn({nPart * samples, 3});
    py::array_t<scalarType> betaReturn({nPart * samples, 3});
    auto positionMutable = positionReturn.mutable_unchecked<2>();
    auto betaMutable = betaReturn.mutable_unchecked<2>();

    for (long unsigned int i = 0; i < nPart * samples; i++)
    {
        scalarType* pos_ptr = m_beamPosition[i].data_ptr<scalarType>();
        scalarType* beta_ptr = m_beamBeta[i].data_ptr<scalarType>();
        positionMutable(i, 0) = pos_ptr[0];
        positionMutable(i, 1) = pos_ptr[1];
        positionMutable(i, 2) = pos_ptr[2];
        betaMutable(i, 0) = beta_ptr[0];
        betaMutable(i, 1) = beta_ptr[1];
        betaMutable(i, 2) = beta_ptr[2];
    }

    // Change shape
    positionReturn.resize({nPart, samples, 3});
    betaReturn.resize({nPart, samples, 3});

    return py::make_tuple(m_time, positionReturn, betaReturn);
}

void Track::setTime(scalarType time_init, scalarType time_end,
    long unsigned int time_steps)
{
    m_time = std::vector<scalarType>(time_steps);
    m_dt = (time_end - time_init) / time_steps;
    for(long unsigned int i = 0; i < time_steps; i++)
    {
        m_time[i] = time_init + i * m_dt;
    }
    m_timeSet = true;
}

void Track::setCentralInit(const vectorType &position_0, 
    const vectorType &momentum_0)
{
    m_initPosition = position_0;
    m_initMomemtum = momentum_0;
    m_initSet = true;
}

void Track::setBeamInit(const py::array_t<scalarType>& position_0, 
    const py::array_t<scalarType>& momentum_0)
{
    py::buffer_info positionInfo = position_0.request();
    auto poisitonPtr = static_cast<scalarType*>(positionInfo.ptr);
    py::buffer_info momentumInfo = momentum_0.request();
    auto momentumPtr = static_cast<scalarType*>(momentumInfo.ptr);
    int n_rows = positionInfo.shape[0];
    // print the shape of position and momentum

    for (int i = 0; i < n_rows; i++)
    {
        m_initBeamPosition.push_back(vectorType(&poisitonPtr[i*3]));
        m_initBeamMomentum.push_back(vectorType(&momentumPtr[i*3]));
    }
    m_beamSet = true;
}

void Track::updateTrack(std::vector<vectorType>& position,
    std::vector<vectorType>& momentum, 
    std::vector<vectorType>& beta, scalarTensor gamma, int index)
{
    vectorType posK1, posK2, posK3, posK4;
    vectorType momK1, momK2, momK3, momK4;
    vectorType momStep;
    
    vectorType field = m_fieldContainer.getField(position[index]);
    posK1 = pushPosition(momentum[index], gamma);
    momK1 = pushMomentum(momentum[index], field, gamma);   
    field = m_fieldContainer.getField(position[index] + posK1 * m_dt / 2.);
    momStep = momentum[index] + momK1 * m_dt / 2.;
    posK2 = pushPosition(momStep, gamma);
    momK2 = pushMomentum(momStep, field, gamma);
    field = m_fieldContainer.getField(position[index] + posK2 * m_dt / 2.);
    momStep = momentum[index] + momK2 * m_dt / 2.;
    posK3 = pushPosition(momStep, gamma);
    momK3 = pushMomentum(momStep, field, gamma);
    field = m_fieldContainer.getField(position[index] + posK3 * m_dt);
    momStep = momentum[index] + momK3 * m_dt;
    posK4 = pushPosition(momStep, gamma);
    momK4 = pushMomentum(momStep, field, gamma);
    position[index+1] = position[index] + (m_dt / 6.) * (posK1 + 2. * posK2
                                                + 2. * posK3 + posK4);
    momentum[index+1] = momentum[index] + (m_dt / 6.) * (momK1 + 2. * momK2
                                                + 2. * momK3 + momK4);

    beta[index+1] = momentum[index+1] / (c_light * gamma);
}

vectorType Track::pushPosition(const vectorType &momentum,
     scalarTensor gamma) const
{
    return momentum / gamma;
}

vectorType Track::pushMomentum(const vectorType &momentum,
    const vectorType &field, scalarTensor gamma) const
{
    return -1 * momentum.Cross(field) / gamma;
}