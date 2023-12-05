#ifndef FieldContainer_HH
#define FieldContainer_HH

#include <vector>

#ifdef USE_TORCH
    #include "TorchVector.hh"
    typedef float scalarType;
    typedef TorchVector vectorType;
#else
    #include "ThreeVector.hh"
    typedef double scalarType;
    typedef ThreeVector vectorType;
#endif

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
    FieldBlock(const vectorType& location, const vectorType& fieldStrength,
        scalarType length, scalarType edgeLength);

    virtual ~FieldBlock(){};

    /**
     * Gets the magnetic field at a given position
     *
     * @param position: Position to calculate the field at.
     * @return Magnetic field three vector.
     */
    virtual vectorType getField(const vectorType& position) const = 0;


protected:
    /**
     * Gets the decay factor of the field at a distance z from the magnet bulk
     *
     * @param z: Distance outside main part of magnet
     * @return Factor by which the field decays
     */
    scalarType getEdge(scalarType z) const;

    vectorType m_location;         // Centre point of magnet
    vectorType m_fieldStrength;    // Field strength vector
    scalarType m_length;                // Length of magnet
    scalarType m_edgeLength;            // Magnet edge length
    scalarType m_edgeScaleFact;         // Edge scaling factor
    scalarType m_constLengthHalf;       // Half of the length where field is const

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
    void addElement(int order, const vectorType& location,
        const vectorType& fieldStrength, scalarType length,
        scalarType edgeLength);


    /**
     * Gets the total magnetic field at a given position
     *
     * @param position: Position to calculate the field at.
     * @return Magnetic field three vector.
     */
    vectorType getField(const vectorType& position) const;

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
    Dipole(const vectorType& location, const vectorType& fieldStrength,
        scalarType length, scalarType edgeLength);

    ~Dipole(){};

    /**
     * Gets the magnetic field at a given position. Will be m_fieldStrength
     * inside the bulk of the magnet and m_fieldStrength * decay factor outside.
     *
     * @param position: Position to calculate the field at.
     * @return Dipole field three vector.
     */
    vectorType getField(const vectorType& position) const override;

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
    Quadrupole(const vectorType& location, const vectorType& fieldStrength,
        scalarType length, scalarType edgeLength);

    ~Quadrupole(){};

    /**
     * Gets the magnetic field at a given position. Will be m_fieldStrength
     * inside the bulk of the magnet and m_fieldStrength * decay factor outside.
     *
     * @param position: Position to calculate the field at.
     * @return Quadrupole field three vector.
     */
    vectorType getField(const vectorType& position) const override;
};

#endif