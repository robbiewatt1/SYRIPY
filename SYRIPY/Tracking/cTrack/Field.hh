#ifndef FieldContainer_HH
#define FieldContainer_HH

#include <vector>
#include "ThreeVector.hh"


class FieldBlock
{
public:

    FieldBlock(){};
    
    /**
     * Constructor used to define field element.
     * 
     * @param location: Centre location of magnet.
     * @param fieldStrength: Field strength vector. B for dipole or db/dr for 
     *      quadrupole
     * @param length: Length of the magnet in the z dimension
     * @param edgeLength: Length for the magnetic field to decay to 10% outside
     *      bulk of the magnet
     */
    FieldBlock(const ThreeVector& location, const ThreeVector& fieldStrength,
        double length, double edgeLength);

    virtual ~FieldBlock(){};

    /**
     * Gets the magnetic field at a given position
     *
     * @param position: Position to calculate the field at.
     * @return Magnetic field three vector.
     */
    virtual ThreeVector getField(const ThreeVector& position) const = 0;


protected:
    /**
     * Gets the decay factor of the field at a distance z from the magnet bulk
     *
     * @param z: Distance outside main part of magnet
     * @return Factor by which the field decays
     */
    double getEdge(double z) const;

    ThreeVector m_location;         // Centre point of magnet
    ThreeVector m_fieldStrength;    // Field strength vector
    double m_length;                // Length of magnet
    double m_edgeLength;            // Magnet edge length
    double m_edgeScaleFact;         // Edge scaling factor
    double m_constLengthHalf;       // Half of the length where field is const
};


class FieldContainer
{
public:
    /*
     * Default constructor
     */
    FieldContainer(){};

    /**
     * Constructor used if vector of elements already exists
     * 
     * @param fieldContainer: Vector of field elements
     */
    FieldContainer(const std::vector<FieldBlock*>& fieldContainer)
    {
        m_fieldContainer = fieldContainer;
    }

    ~FieldContainer(){};

    /**
     * Adds elements (dipole or quadrupole) to the field container.
     *
     * @param order: Field order, 1 = Dipole, 2 = Quad
     * @param location: Centre location of magnet.
     * @param fieldStrength: Field strength vector. B for dipole or db/dr for 
     *      quadrupole
     * @param length: Length of the magnet in the z dimension
     * @param edgeLength: Length for the magnetic field to decay to 10% outside
     *      bulk of the magnet
     */
    void addElement(int order, const ThreeVector& location,
        const ThreeVector& fieldStrength, double length, double edgeLength);


    /**
     * Gets the total magnetic field at a given position
     *
     * @param position: Position to calculate the field at.
     * @return Magnetic field three vector.
     */
    ThreeVector getField(const ThreeVector& position) const;

private:
    std::vector<FieldBlock*> m_fieldContainer;
    
};


class Dipole: public FieldBlock
{
public:
    Dipole(){};
    /**
     * Constructor used to define dipole element.
     * 
     * @param location: Centre location of dipole.
     * @param fieldStrength: Field strength vector.
     * @param length: Length of the magnet in the z dimension
     * @param edgeLength: Length for the magnetic field to decay to 10% outside
     *      bulk of the magnet
     */
    Dipole(const ThreeVector& location, const ThreeVector& fieldStrength,
        double length, double edgeLength);

    ~Dipole(){};

    /**
     * Gets the magnetic field at a given position. Will be m_fieldStrength
     * inside the bulk of the magnet and m_fieldStrength * decay factor outside.
     *
     * @param position: Position to calculate the field at.
     * @return Dipole field three vector.
     */
    ThreeVector getField(const ThreeVector& position) const override;

};


class Quadrupole: public FieldBlock
{
public:
    Quadrupole(){}
    /**
     * Constructor used to define quadrupole element.
     * 
     * @param location: Centre location of quadrupole.
     * @param fieldStrength: Gradient field strength vector.
     * @param length: Length of the magnet in the z dimension
     * @param edgeLength: Length for the magnetic field to decay to 10% outside
     *      bulk of the magnet
     */
    Quadrupole(const ThreeVector& location, const ThreeVector& fieldStrength,
        double length, double edgeLength);

    ~Quadrupole(){};

    /**
     * Gets the magnetic field at a given position. Will be m_fieldStrength
     * inside the bulk of the magnet and m_fieldStrength * decay factor outside.
     *
     * @param position: Position to calculate the field at.
     * @return Quadrupole field three vector.
     */
    ThreeVector getField(const ThreeVector& position) const override;
};

#endif