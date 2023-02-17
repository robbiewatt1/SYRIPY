#include "Track.hh"
#include "ThreeVector.hh"
#include <fstream>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <random>



namespace py = pybind11;

py::tuple Track::simulateTrack()
{
    std::vector<ThreeVector> position = std::vector<ThreeVector>(m_time.size());
    std::vector<ThreeVector> momentum = std::vector<ThreeVector>(m_time.size());
    std::vector<ThreeVector> beta = std::vector<ThreeVector>(m_time.size());

    position[0] = m_initPosition;
    momentum[0] = c_light * std::sqrt(m_gamma * m_gamma - 1.)
        * m_initDirection.Norm();
    beta[0] = momentum[0] / std::sqrt(c_light * c_light + momentum[0].Mag2());

    for(long unsigned int i = 0; i < m_time.size()-1; i++)
    {
        updateTrack(position, momentum, beta, i);
    }

    // Convert to numpy and return
    int samples = m_time.size();
    py::array_t<double> time_p = py::cast(m_time);
    std::vector<double> position_c(3 * samples);
    for(int i = 0; i < samples; i++)
    {
        position_c[3*i] = position[i][0];
        position_c[3*i+1] = position[i][1];
        position_c[3*i+2] = position[i][2];
    }
    py::array_t<double> position_p = py::cast(position_c);
    position_p.resize({samples, 3});
    std::vector<double> beta_c(3 * samples);
    for(int i = 0; i < samples; i++)
    {
        beta_c[3*i] = beta[i][0];
        beta_c[3*i+1] = beta[i][1];
        beta_c[3*i+2] = beta[i][2];
    }
    py::array_t<double> beta_p = py::cast(beta_c);
    beta_p.resize({samples, 3});
    return py::make_tuple(time_p, position_p, beta_p);
}

void Track::setTime(double time_init, double time_end, 
    long unsigned int time_steps)
{
    m_time = std::vector<double>(time_steps);
    m_dt = (time_end - time_init) / time_steps;
    for(long unsigned int i = 0; i < time_steps; i++)
    {
        m_time[i] = time_init + i * m_dt;
    }
}

void Track::setCentralInit(const ThreeVector &position_0, 
    const ThreeVector &direction_0, double gamma)
{
    m_initPosition = position_0;
    m_initDirection = direction_0.Norm();
    m_gamma = gamma;
}

void Track::setBeamParams(py::array_t<double>& moments)
{
    auto m = moments.unchecked<1>(); 
    for (int i = 0; i < m.size(); ++i)
    {
        std::cout << m(i) << std::endl;
    }
}

void Track::updateTrack(std::vector<ThreeVector>& position,
    std::vector<ThreeVector>& momentum, std::vector<ThreeVector>& beta,
    int index)
{
        ThreeVector posK1, posK2, posK3, posK4;
        ThreeVector momK1, momK2, momK3, momK4;
        ThreeVector field;

        field = m_fieldCont.getField(position[index]);
        posK1 = pushPosition(momentum[index]);
        momK1 = pushMomentum(momentum[index], field);   
        field = m_fieldCont.getField(position[index] + posK1 * m_dt / 2.);
        posK2 = pushPosition(momentum[index] + momK1 * m_dt / 2.);
        momK2 = pushMomentum(momentum[index] + momK1 * m_dt / 2., field);
        field = m_fieldCont.getField(position[index] + posK2 * m_dt / 2.);
        posK3 = pushPosition(momentum[index] + momK2 * m_dt / 2.);
        momK3 = pushMomentum(momentum[index] + momK2 * m_dt / 2., field);
        field = m_fieldCont.getField(position[index] + posK3 * m_dt);
        posK4 = pushPosition(momentum[index] + momK3 * m_dt);
        momK4 = pushMomentum(momentum[index] + momK3 * m_dt, field);

        position[index+1] = position[index] + (m_dt / 6.) * (posK1 + 2.0 * posK2
                                                    + 2.0 * posK3 + posK4);
        momentum[index+1] = momentum[index] + (m_dt / 6.) * (momK1 + 2.0 * momK2
                                                    + 2.0 * momK3 + momK4);
        beta[index+1] = momentum[index+1] / std::sqrt(c_light * c_light
            + momentum[index+1].Mag2());
}

ThreeVector Track::pushPosition(const ThreeVector &momentum) const
{
    double gamma  = std::sqrt(1.0 + momentum.Mag2() / (c_light * c_light));
    return momentum / gamma;
}


ThreeVector Track::pushMomentum(const ThreeVector &momentum, 
        const ThreeVector &field) const
{
    double gamma  = std::sqrt(1.0 + momentum.Mag2() / (c_light * c_light));
    return -1 * momentum.Cross(field) / gamma;
}
