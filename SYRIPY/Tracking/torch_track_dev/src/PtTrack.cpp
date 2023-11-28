#include "PtTrack.hh"
#include <fstream>
#include <stdexcept>
#include <pybind11/stl.h>
#include <torch/script.h>


#ifdef _OPENMP
    #include <omp.h>
#endif

py::tuple PtTrack::test()
{
    vectorType position = vectorType({0., 0., 0.}, true);
    vectorType momentum = vectorType({0., 1., 0.}, true);
    Dipole field = Dipole(vectorType({0., 0., 0.}),
        vectorType({0., 0., 5.}), 1.e6, 1.);
    FieldContainer field_con = FieldContainer(std::vector<FieldBlock*>{&field});
    setField(field_con);
    setTime(0., 1., 1000);
    setCentralInit(position, momentum);
    py::tuple result = simulateTrack();
    auto gradients = backwardTrack();
    return py::make_tuple(result, gradients);
}

py::tuple PtTrack::simulateTrack()
{
    if (!m_timeSet || !m_initSet || !m_fieldSet)
    {
        throw std::runtime_error("Error: Time or initial conditions not set.");
    }

    m_position = std::vector<vectorType>(m_time.size());
    m_momentum = std::vector<vectorType>(m_time.size());
    m_beta = std::vector<vectorType>(m_time.size());

    m_position[0] = m_initPosition;
    m_momentum[0] = m_initMomemtum;
    m_beta[0] = m_momentum[0] / math::sqrt(c_light * c_light
        + m_momentum[0].Magnitude2());

    for (long unsigned int i = 0; i < m_time.size() - 1; i++)
    {
        updateTrack(m_position, m_momentum, m_beta, i);
    }

    // Convert and return
    int samples = m_time.size();
    py::array_t<scalarType> positionReturn({samples, 3});
    py::array_t<scalarType> betaReturn({samples, 3});

    for (long unsigned int i = 0; i < m_time.size(); i++)
    {
        scalarType* pos_ptr = m_position[i].data_ptr<scalarType>();
        scalarType* beta_ptr = m_beta[i].data_ptr<scalarType>();
        positionReturn.mutable_at(i, 0) = pos_ptr[0];
        positionReturn.mutable_at(i, 1) = pos_ptr[1];
        positionReturn.mutable_at(i, 2) = pos_ptr[2];
        betaReturn.mutable_at(i, 0) = beta_ptr[0];
        betaReturn.mutable_at(i, 1) = beta_ptr[1];
        betaReturn.mutable_at(i, 2) = beta_ptr[2];
    }
    return py::make_tuple(m_time, positionReturn, betaReturn);
}

py::tuple PtTrack::backwardTrack(std::vector<vectorType> &grad_outputs)
{
    // Check that USE_TORCH is defined
#ifndef USE_TORCH
    throw std::runtime_error("Error: Can't run backwardTrack without"
        "libtorch.");
#endif

    // check that forward track has been run
    if (m_position.size() == 0)
    {
        throw std::runtime_error("Error: Can't run backward through track"
            "without running forward first.");
    }
     // Convert to vector of Tensors
    std::vector<torch::Tensor> tensor_position(m_position.begin(),
        m_position.end());
    std::vector<torch::Tensor> tensor_momenutm(m_momentum.begin(),
        m_momentum.end());
    std::vector<torch::Tensor> tensor_cat = tensor_position;
    tensor_cat.insert(tensor_cat.end(), tensor_momenutm.begin(),
        tensor_momenutm.end());

    auto gradients = torch::autograd::grad(tensor_cat,
        {m_initPosition, m_initMomemtum}, grad_outputs);

    // Need to check if both params require grads or just 1
    py::array_t<scalarType> position_grad = py::array_t<scalarType>(3,
        gradients[0].data_ptr<scalarType>());
    py::array_t<scalarType> momentum_grad = py::array_t<scalarType>(3,
        gradients[1].data_ptr<scalarType>());

    return py::make_tuple(position_grad, momentum_grad);
}

void PtTrack::simulateBeam(int nPart)
{
    if (!m_timeSet || !m_beamSet || !m_fieldSet)
    {
        std::cerr << "Error: Time or initial conditions not set." << std::endl;
        return;
    }
    int tSamp = m_time.size();
    std::vector<vectorType> position = std::vector<vectorType>(nPart * tSamp);
    std::vector<vectorType> momentum = std::vector<vectorType>(nPart * tSamp);
    std::vector<vectorType> beta = std::vector<vectorType>(nPart * tSamp);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < nPart; i++)
    {
        int partIndex = i * tSamp;
        position[partIndex] = m_initBeamPosition[i];
        momentum[partIndex] = m_initBeamMomemtum[i];
        beta[partIndex] = momentum[partIndex] / math::sqrt(c_light * c_light
            + momentum[partIndex].Magnitude2());
        for (long unsigned int j = 0; j < tSamp - 1; j++)
        {
            updateTrack(position, momentum, beta, partIndex + j);
        }
    }
}

void PtTrack::setTime(scalarType time_init, scalarType time_end,
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

void PtTrack::setCentralInit(const vectorType &position_0, 
    const vectorType &momentum_0)
{
    m_initPosition = position_0;
    m_initMomemtum = momentum_0;
    m_initSet = true;
}

void PtTrack::setBeamInit(const std::vector<vectorType> &position_0, 
    const std::vector<vectorType> &momentum_0)
{
    m_initBeamPosition = position_0;
    m_initBeamMomemtum = momentum_0;
    m_beamSet = true;
}

void PtTrack::updateTrack(std::vector<vectorType>& position,
    std::vector<vectorType>& momentum, 
    std::vector<vectorType>& beta, int index)
{
    vectorType posK1, posK2, posK3, posK4;
    vectorType momK1, momK2, momK3, momK4;


    vectorType field = m_fieldContainer.getField(position[index]);
    posK1 = pushPosition(momentum[index]);
    momK1 = pushMomentum(momentum[index], field);   
    field = m_fieldContainer.getField(position[index] + posK1 * m_dt / 2.);
    posK2 = pushPosition(momentum[index] + momK1 * m_dt / 2.);
    momK2 = pushMomentum(momentum[index] + momK1 * m_dt / 2., field);
    field = m_fieldContainer.getField(position[index] + posK2 * m_dt / 2.);
    posK3 = pushPosition(momentum[index] + momK2 * m_dt / 2.);
    momK3 = pushMomentum(momentum[index] + momK2 * m_dt / 2., field);
    field = m_fieldContainer.getField(position[index] + posK3 * m_dt);
    posK4 = pushPosition(momentum[index] + momK3 * m_dt);
    momK4 = pushMomentum(momentum[index] + momK3 * m_dt, field);
    position[index+1] = position[index] + (m_dt / 6.) * (posK1 + 2. * posK2
                                                + 2. * posK3 + posK4);
    momentum[index+1] = momentum[index] + (m_dt / 6.) * (momK1 + 2. * momK2
                                                + 2. * momK3 + momK4);
                                                
    beta[index+1] = momentum[index+1] / math::sqrt(c_light * c_light
        + momentum[index+1].Magnitude2()); 
}

vectorType PtTrack::pushPosition(const vectorType &momentum) const
{
    // get the magnitude of the momentum
    return momentum / math::sqrt(1. + momentum.Magnitude2()
        / (c_light * c_light));
}

vectorType PtTrack::pushMomentum(const vectorType &momentum,
    const vectorType &field) const
{
    return -1 * momentum.Cross(field) / (math::sqrt(1.0 + momentum.Magnitude2()
         / (c_light * c_light)));
}