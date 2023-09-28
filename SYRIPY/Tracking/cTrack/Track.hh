#ifndef TRACK_HH
#define TRACK_HH

#include "ThreeVector.hh"
#include "ThreeMatrix.hh"
#include "Field.hh"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>
#include <random>


namespace py = pybind11;

class Track
{
public:

    /**
     * Default constructor. All this does is set the initialise switches.
     */
    Track(){m_timeSet = false; m_initSet = false;};

    ~Track(){};

    /**
     * Main function for simulating the track of a single electron. Requires the
     * direction / position / time information to be set before running. Returns
     * track data.
     * @return Tuple of np.arrays (time, position, beta) 
     */
    py::tuple simulateTrack();

    /**
     * Main function for simulating multiple electrons (i.e. beam). Requires the
     * direction / position / time / beam param information to be set before
     * running. Updates the m_position / m_momentum / m_beta vectors.
     */
    py::tuple simulateBeam(unsigned int nPart);

    /**
     * Sets the time parameters for tracking.
     * 
     * @param time_init: Initial time of the track.
     * @param time_end: End time of the track.
     * @param time_steps: Number of integration steps.
     */
    void setTime(double time_init, double time_end, 
        long unsigned int time_steps);

    /**
     * Sets the initial parameters of the central track, i.e. the position,
     * direction and gamma factor.
     *
     * @param position_0: Initial position of the track.
     * @param direction_0: Initial direction of the track
     * @param gamma: Lorentz factor of electron.
     */
    void setCentralInit(const ThreeVector &position_0,
        const ThreeVector &direction_0, double gamma);

    /**
     * Sets the second order moments of the beam distribution function. We 
     * assume no correlation in x / y / energy. The elements of moments are:
     * m[0] = s_x^2
     * m[1] = s_x s_x'
     * m[2] = s_x'^2
     * m[3] = s_y^2
     * m[4] = s_y s_y'
     * m[5] = s_y'^2
     * m[6] = s_g^2
     *
     * @param moments: Numpy array of second order moments
     */
    void setBeamParams(py::array_t<double>& moments);

    /**
     * Sets the field container for the solver.
     * @param fieldCont: Instance of field container class
     */    
    void setField(const FieldContainer& fieldCont){m_fieldCont = fieldCont;}

    /**
     * Returns the track time / position / beta as a tuple
     */

private:

    /**
     * Performs a single update set using an rk4 method. 
     *
     * @param position: Previous position of the electron.
     * @param momentum: Previous momentum of the electron.
     * @param dt: Time step for integration.
     * @param index: Current step index of integration. 
     */
    void updateTrack(std::vector<ThreeVector>& position,
        std::vector<ThreeVector>& momentum, std::vector<ThreeVector>& beta, 
        int index);

    /**
     * Used to update the position of the electron.
     *
     * @param momentum: Current momentum of the electron.
     * @return Gradient of the position.
     */
    ThreeVector pushPosition(const ThreeVector &momentum) const;

    /**
     * Used to update the momentum of the electron.
     *
     * @param momentum: Current momentum of the electron.
     * @param field: Current magnetic field experienced by the electron.
     * @return Gradient of the momentum.
     */
    ThreeVector pushMomentum(const ThreeVector &momentum, 
        const ThreeVector &field) const;

    // Initial central parameters
    double m_gamma;
    ThreeVector m_initPositionCent;
    ThreeVector m_initDirectionCent;

    // Time params
    double m_dt;
    std::vector<double> m_time;

    // Beam params and rotation
    std::vector<double> m_beamParams;
    ThreeMatrix m_rotation;


    // Container class of magnetic field
    FieldContainer m_fieldCont;

    // Bools to test if setup is run
    bool m_timeSet, m_initSet;

    static constexpr double c_light = 0.299792458;
};


struct normal_random_variable
{
    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, 
        Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() 
            * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{
            mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};

#endif