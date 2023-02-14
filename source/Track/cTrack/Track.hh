#ifndef TRACK_HH
#define TRACK_HH

#include "ThreeVector.hh"
#include "Field.hh"
#include <vector>

class Track
{
public:

    /**
     * Defult constructor. All this does is set the initiliser swtiches.
     */
    Track(){m_timeSet = false; m_initSet = false;};

    ~Track(){};

    /**
     * Main function, Simulates the track of an electron. Requires the
     * direction / poition / time information to be set before running. Updates
     * the m_position / m_monetum / m_beta vectors.
     */
    void simulateTrack();

    /**
     * Sets the time paramters for tracking.
     * 
     * @param time_init: Initial time of the track.
     * @param time_end: End time of the track.
     * @param time_steps: Number of intergation steps.
     */
    void setTime(double time_init, double time_end, 
        long unsigned int time_steps);

    /**
     * Sets the initial paramters of the track, i.e. the position, direction
     * and gamma factor.
     *
     * @param position_0: Initial position of the track.
     * @param direction_0: Initial direction of the track
     * @param gamma: Lorentz factor of electron.
     */
    void setInit(const ThreeVector &position_0, const ThreeVector &direction_0,
        double gamma);

    // void save(std::string name);

private:

    /**
     * Performs a single update set using an rk4 method. 
     *
     * @param position: Previous position of the electron.
     * @param momentum: Previous momeumtum of the electron.
     * @param dt: Time step for integration.
     * @param index: Current step index of integration. 
     */
    void updateTrack(const ThreeVector &position, const ThreeVector &momentum,
        double dt, int index);

    /**
     * Used to update the position of the electron.
     *
     * @param momentum: Current momeumtum of the electron.
     * @return Gradient of the position.
     */
    ThreeVector pushPosition(const ThreeVector &momentum) const;

    /**
     * Used to update the momentum of the electron.
     *
     * @param momentum: Current momeumtum of the electron.
     * @param field: Current magnetic field experienced by the electron.
     * @return Gradient of the mometum.
     */
    ThreeVector pushMomentum(const ThreeVector &momentum, 
        const ThreeVector &field) const;

    // Initial electron param,ters
    double m_gamma;
    ThreeVector m_initPosition;
    ThreeVector m_initDirection;

    // Electron track vectors
    std::vector<double> m_time;
    std::vector<ThreeVector> m_position;
    std::vector<ThreeVector> m_momentum;
    std::vector<ThreeVector> m_beta;

    // Container class of magnetic field
    FieldContainer* m_fieldCon;

    // Bools to test if setup is run
    bool m_timeSet, m_initSet;

    static constexpr double c_light = 0.299792458;
};


#endif