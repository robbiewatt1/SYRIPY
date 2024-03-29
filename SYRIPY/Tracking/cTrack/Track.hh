#ifndef TRACK_HH
#define TRACK_HH

#include "Field.hh"
#include <vector>

#include <pybind11/numpy.h>

#ifdef USE_TORCH
    #include "TorchVector.hh"
    typedef float scalarType;
    typedef TorchVector vectorType;
    typedef torch::Tensor scalarTensor;
    namespace math = torch;
#else
    #include "ThreeVector.hh"
    typedef double scalarType;
    typedef double scalarTensor;
    typedef ThreeVector vectorType;
    namespace math = std;
#endif

namespace py = pybind11;

/*
Class to track a particle through the magnetic setup. The base vector class can
be eithe torch or my simple 3 vector.
*/


class Track
{
public:
    /* Defult constructor which sets setup switches*/
    Track(){m_timeSet = false; m_initSet = false;};
    
    ~Track(){};

    /**
     * Main function for simulating the track of a single electron. Requires the
     * momentum / position / time information to be set before running. Returns
     * track data.
     * @return Tuple of torch (time, position, beta) 
     */
    py::tuple simulateTrack();

    /**
     * Function to calculate graident of track. Requires forward track to be run
     * first.
     * @return Tuple of torch (position_grad, beta_grad) 
     */
    py::tuple backwardTrack(py::array_t<scalarType> &positionGrad,
        py::array_t<scalarType> &betaGrad);

    /**
     * Main function for simulating multiple electrons (i.e. beam). Requires the
     * momentum / position / time / beam param information to be set before
     * running.
     */
    py::tuple simulateBeam();

    /**
     * Sets the time parameters for tracking.
     * 
     * @param time_init: Initial time of the track.
     * @param time_end: End time of the track.
     * @param time_steps: Number of integration steps.
     */
    void setTime(scalarType time_init, scalarType time_end,
        long unsigned int time_steps);

    /**
     * Sets the initial parameters of the track, i.e. the position,
     * and momentum.
     *
     * @param position_0: Initial position of the track.
     * @param momentum_0: Initial direction of the track
     */
    void setCentralInit(const vectorType &position_0, 
        const vectorType &momentum_0);

    /**
     * Sets the initial parameters of the beam, i.e. the position,
     * momentum.
     *
     * @param position_0: Initial position of the track.
     * @param momentum_0: Initial direction of the track
     */
    void setBeamInit(const py::array_t<scalarType> &position_0, 
        const py::array_t<scalarType> &momentum_0);

    /**
     * Sets the field container for the solver.
     * @param fieldCont: Instance of field container class
     */    
    void setField(const FieldContainer& fieldCont){
        m_fieldContainer = fieldCont; m_fieldSet = true;};


private:

    /**
     * Performs a single update set using an rk4 method. 
     *
     * @param position: Previous positions of the electron.
     * @param momentum: Previous momentums of the electron.
     * @param beta: Previous beta values of the electron.
     * @param index: Current step index of integration. 
     */
    void updateTrack(std::vector<vectorType>& position,
        std::vector<vectorType>& momentum, std::vector<vectorType>& beta,
        scalarTensor gamma, int index);


    /**
     * Used to update the position of the electron.
     *
     * @param momentum: Current momentum of the electron.
     * @return Gradient of the position.
     */
    vectorType pushPosition(const vectorType &momentum, scalarTensor gamma
        ) const;

    /**
     * Used to update the momentum of the electron.
     *
     * @param momentum: Current momentum of the electron.
     * @param field: Current magnetic field experienced by the electron.
     * @return Gradient of the momentum.
     */
    vectorType pushMomentum(const vectorType &momentum,
        const vectorType &field, scalarTensor gamma) const;


    // Initial single conditions
    vectorType m_initMomemtum;
    vectorType m_initPosition;

    // Saved track data
    std::vector<vectorType> m_position;
    std::vector<vectorType> m_momentum;
    std::vector<vectorType> m_beta;

    // Initial beam conditions
    std::vector<vectorType> m_initBeamPosition;
    std::vector<vectorType> m_initBeamMomentum;

    // Saved beam data
    std::vector<vectorType> m_beamPosition;
    std::vector<vectorType> m_beamMomentum;
    std::vector<vectorType> m_beamBeta;

    // Time paramters
    scalarType m_dt;
    std::vector<scalarType> m_time;

    // Field class
    FieldContainer m_fieldContainer;

    // Setup switches
    bool m_timeSet, m_initSet, m_fieldSet, m_beamSet;

    static constexpr scalarType c_light = 0.299792458;
};

#endif