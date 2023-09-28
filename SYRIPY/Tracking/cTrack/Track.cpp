#include "Track.hh"
#include "ThreeVector.hh"
#include <fstream>
#include <pybind11/stl.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

py::tuple Track::simulateTrack()
{
    std::vector<ThreeVector> position = std::vector<ThreeVector>(m_time.size());
    std::vector<ThreeVector> momentum = std::vector<ThreeVector>(m_time.size());
    std::vector<ThreeVector> beta = std::vector<ThreeVector>(m_time.size());

    position[0] = m_initPositionCent;
    momentum[0] = c_light * std::sqrt(m_gamma * m_gamma - 1.)
        * m_initDirectionCent.Norm();
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
        position_c[3*i]   = position[i][0];
        position_c[3*i+1] = position[i][1];
        position_c[3*i+2] = position[i][2];
    }
    py::array_t<double> position_p = py::cast(position_c);
    position_p.resize({samples, 3});
    std::vector<double> beta_c(3 * samples);
    for(int i = 0; i < samples; i++)
    {
        beta_c[3*i]   = beta[i][0];
        beta_c[3*i+1] = beta[i][1];
        beta_c[3*i+2] = beta[i][2];
    }
    py::array_t<double> beta_p = py::cast(beta_c);
    beta_p.resize({samples, 3});
    return py::make_tuple(time_p, position_p, beta_p);
}

py::tuple Track::simulateBeam(int nPart)
{
    int samples = m_time.size();
    std::vector<ThreeVector> position 
        = std::vector<ThreeVector>(nPart * samples);
    std::vector<ThreeVector> momentum 
        = std::vector<ThreeVector>(nPart * samples);
    std::vector<ThreeVector> beta 
        = std::vector<ThreeVector>(nPart * samples);

    // Get the rotation matrix
    m_rotation = m_initDirectionCent.RotateToAxis(
            ThreeVector(0, 0, 1)).Inverse();

    // Setup the particle params generators
    Eigen::MatrixXd x_covar(2, 2);
    x_covar << m_beamParams[0], m_beamParams[1],
               m_beamParams[1], m_beamParams[2];
    normal_random_variable x_gen { x_covar };
    Eigen::MatrixXd y_covar(2, 2);
    y_covar << m_beamParams[3], m_beamParams[4],
               m_beamParams[4], m_beamParams[5];
    normal_random_variable y_gen { y_covar };
    Eigen::MatrixXd g_covar(1, 1);
    g_covar << m_beamParams[7];
    normal_random_variable g_gen { g_covar };


#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < nPart; ++i)
    {
        int partIndex = i * samples;
        Eigen::VectorXd x_sample = x_gen();
        Eigen::VectorXd y_sample = y_gen();
        Eigen::VectorXd g_sample = g_gen();

        // Local initial position
        ThreeVector init_position = ThreeVector(x_sample[0],
                                                y_sample[0],
                                                0);
        // Local initial direction
        double alpha = std::sqrt(1 + std::tan(x_sample[1]) 
            * std::tan(x_sample[1]) + std::tan(y_sample[1])
            * std::tan(y_sample[1]));
        ThreeVector init_direction = ThreeVector(std::tan(x_sample[1]) / alpha,
                                                 std::tan(y_sample[1]) / alpha,
                                                 1 / alpha);

        // Rotate direction and position  to main axis
        init_position = m_rotation * init_position + m_initPositionCent;
        init_direction = m_rotation * init_direction;

        double gamma = m_gamma + g_sample[0];
                
        // Fill the initial point
        position[partIndex] = init_position;
        momentum[partIndex] = c_light * std::sqrt(gamma * gamma - 1.)
            * init_direction.Norm();
        beta[partIndex] = momentum[partIndex] / std::sqrt(c_light * c_light
            + momentum[partIndex].Mag2());

        for(long unsigned int j = 0; j < m_time.size()-1; j++)
        {
            updateTrack(position, momentum, beta, partIndex+j);
        }

    }

    // prepare output
    py::array_t<double> time_p = py::cast(m_time);
    // Move data to buffer
    py::array_t<double> position_p(nPart * samples * 3);
    py::array_t<double> beta_p(nPart * samples * 3);

    auto r = position_p.mutable_unchecked<1>();
    auto b = beta_p.mutable_unchecked<1>();

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < nPart * samples; i++)
    {
        r[3*i]   = position[i][0];
        r[3*i+1] = position[i][1];
        r[3*i+2] = position[i][2];
        b[3*i]   = beta[i][0];
        b[3*i+1] = beta[i][1];
        b[3*i+2] = beta[i][2];
    }

    // Change shape
    position_p.resize({nPart, samples, 3});
    beta_p.resize({nPart, samples, 3});

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
    m_initPositionCent = position_0;
    m_initDirectionCent = direction_0.Norm();
    m_gamma = gamma;
}

void Track::setBeamParams(py::array_t<double>& moments)
{
    auto m = moments.unchecked<1>();
    if (m.size() != 8)
    {
        std::cerr << "Error: wrong number of beam parameters" << std::endl;
    }
    m_beamParams = std::vector<double>(8);

    for (int i = 0; i < m.size(); ++i)
    {
        m_beamParams[i] = m[i];
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

        position[index+1] = position[index] + (m_dt / 6.) * (posK1 + 2. * posK2
                                                    + 2. * posK3 + posK4);
        momentum[index+1] = momentum[index] + (m_dt / 6.) * (momK1 + 2. * momK2
                                                    + 2. * momK3 + momK4);
        beta[index+1] = momentum[index+1] / std::sqrt(c_light * c_light
            + momentum[index+1].Mag2());
}

ThreeVector Track::pushPosition(const ThreeVector &momentum) const
{
    double gamma  = std::sqrt(1. + momentum.Mag2() / (c_light * c_light));
    return momentum / gamma;
}


ThreeVector Track::pushMomentum(const ThreeVector &momentum, 
        const ThreeVector &field) const
{
    double gamma  = std::sqrt(1.0 + momentum.Mag2() / (c_light * c_light));
    return -1 * momentum.Cross(field) / gamma;
}
