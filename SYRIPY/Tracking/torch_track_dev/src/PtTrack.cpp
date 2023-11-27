#include "PtTrack.hh"
#include <fstream>

#ifdef _OPENMP
    #include <omp.h>
#endif

void PtTrack::simulateTrack()
{
    if (!m_timeSet || !m_initSet || !m_fieldSet)
    {
        std::cerr << "Error: Time or initial conditions not set." << std::endl;
        return;
    }

    std::vector<vectorType> position = std::vector<vectorType>(m_time.size());
    std::vector<vectorType> momentum = std::vector<vectorType>(m_time.size());
    std::vector<vectorType> beta = std::vector<vectorType>(m_time.size());

    position[0] = m_initPosition;
    momentum[0] = m_initMomemtum;
    beta[0] = momentum[0] / math::sqrt(c_light * c_light
        + momentum[0].Magnitude2());

    for (long unsigned int i = 0; i < m_time.size() - 1; i++)
    {
        updateTrack(position, momentum, beta, i);
    }
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
    // 
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