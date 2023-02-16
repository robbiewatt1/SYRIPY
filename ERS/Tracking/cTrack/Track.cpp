#include "Track.hh"
#include "ThreeVector.hh"
#include <fstream>
#include <pybind11/stl.h>

namespace py = pybind11;

void Track::simulateTrack()
{
    m_position = std::vector<ThreeVector>(m_time.size());
    m_momentum = std::vector<ThreeVector>(m_time.size());
    m_beta = std::vector<ThreeVector>(m_time.size());

    m_position[0] = m_initPosition;
    m_momentum[0] = c_light * std::sqrt(m_gamma * m_gamma - 1.)
        * m_initDirection.Norm();
    m_beta[0] = std::sqrt(1. - 1. / (m_gamma * m_gamma)) 
        * m_initDirection.Norm();

    double dt = m_time[1] - m_time[0]
;    for(long unsigned int i = 1; i < m_time.size(); i++)
    {
        updateTrack(m_position[i-1], m_momentum[i-1], dt, i);
    }
}

void Track::setTime(double time_init, double time_end, 
    long unsigned int time_steps)
{
    m_time = std::vector<double>(time_steps);
    double delta_t = (time_end - time_init) / time_steps;
    for(long unsigned int i = 0; i < time_steps; i++)
    {
        m_time[i] = time_init + i * delta_t;
    }
}

void Track::setInit(const ThreeVector &position_0, 
    const ThreeVector &direction_0, double gamma)
{
    m_initPosition = position_0;
    m_initDirection = direction_0.Norm();
    m_gamma = gamma;
}

void Track::updateTrack(const ThreeVector &position,
    const ThreeVector &momentum, double dt, int index)
{
        ThreeVector posK1, posK2, posK3, posK4;
        ThreeVector momK1, momK2, momK3, momK4;
        ThreeVector field;

        field = m_fieldCont.getField(position);
        posK1 = pushPosition(momentum);
        momK1 = pushMomentum(momentum, field);   
        field = m_fieldCont.getField(position + posK1 * dt / 2.);
        posK2 = pushPosition(momentum + momK1 * dt / 2.);
        momK2 = pushMomentum(momentum + momK1 * dt / 2., field);
        field = m_fieldCont.getField(position + posK2 * dt / 2.);
        posK3 = pushPosition(momentum + momK2 * dt / 2.);
        momK3 = pushMomentum(momentum + momK2 * dt / 2., field);
        field = m_fieldCont.getField(position + posK3 * dt);
        posK4 = pushPosition(momentum + momK3 * dt);
        momK4 = pushMomentum(momentum + momK3 * dt, field);

        m_position[index] = position + (dt / 6.) * (posK1 + 2.0 * posK2
                                                    + 2.0 * posK3 + posK4);
        m_momentum[index] = momentum + (dt / 6.) * (momK1 + 2.0 * momK2
                                                    + 2.0 * momK3 + momK4);
        m_beta[index] = m_momentum[index] / std::sqrt(c_light * c_light
            + m_momentum[index].Mag2());
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

py::tuple Track::getTrack() const
{

    int samples = m_time.size();
    py::array_t<double> time_p = py::cast(m_time);
    std::vector<double> position_c(3 * samples);
    for(int i = 0; i < samples; i++)
    {
        position_c[3*i] = m_position[i][0];
        position_c[3*i+1] = m_position[i][1];
        position_c[3*i+2] = m_position[i][2];
    }
    py::array_t<double> position_p = py::cast(position_c);
    position_p.resize({samples, 3});
    std::vector<double> beta_c(3 * samples);
    for(int i = 0; i < samples; i++)
    {
        beta_c[3*i] = m_beta[i][0];
        beta_c[3*i+1] = m_beta[i][1];
        beta_c[3*i+2] = m_beta[i][2];
    }
    py::array_t<double> beta_p = py::cast(beta_c);
    beta_p.resize({samples, 3});
    return py::make_tuple(time_p, position_p, beta_p);
}

/*
void Track::save(std::string name)
{
    std::cout << m_time.size() << std::endl;
    std::ofstream out_r = std::ofstream(name + "_r.txt");
    std::ofstream out_p = std::ofstream(name + "_p.txt");
    for(int i = 0; i < m_time.size(); i++)
    {
        out_r << m_time[i]  << " " << m_position[i][0] << " "
            << m_position[i][1] << " " << m_position[i][2] << "\n";
    }
    for(int i = 0; i < m_time.size(); i++)
    {
        out_p << m_time[i] << " " << m_beta[i][0] << " " << m_beta[i][1] << " "
              << m_beta[i][2] << "\n";
    }
    out_r.close();
    out_p.close();
}
*/