#include "Field.hh"



FieldBlock::FieldBlock(const ThreeVector& location, 
    const ThreeVector& fieldStrength, double length, double edgeLength):
m_location(location), m_fieldStrength(fieldStrength), m_length(length), 
    m_edgeLength(edgeLength)
{
    m_edgeScaleFact = m_edgeLength / 1.4704685172312868;
}

double FieldBlock::getEdge(double z) const
{
    double zp2 = z * z / (m_edgeScaleFact * m_edgeScaleFact);
    return 1. / ((1. + zp2) * (1. + zp2));
}

/* End of FieldBlock
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/

void FieldContainer::addElement(int order, const ThreeVector& location,
        const ThreeVector& fieldStrength, double length, double edgeLength)
{
    FieldBlock element;
    if (order == 1) // dipole
    {
        element = Diploe(location, fieldStrength, length, edgeLength);
    } else if (order == 2)
    {
        element = Quadrupole(location, fieldStrength, length, edgeLength);
    } else
    {
        std::cerr << "Error: Field order must be 1 for dipole or 2 for " 
                     "quadrupole. Instead recieved: " << order << std::endl;
        return;
    }
    m_fieldContainer.push_back(element);
}

ThreeVector FieldContainer::getField(const ThreeVector& position) const
{
    ThreeVector field;
    for (long unsigned int i = 0; i < m_fieldContainer.size(); i++)
    {
        field = field + m_fieldContainer[i].getField(position);
    }
    return field;
}

/* End of FieldContainer
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/

Diploe::Diploe(const ThreeVector& location, const ThreeVector& fieldStrength,
        double length, double edgeLength):
FieldBlock(location, fieldStrength,length, edgeLength)
{
}

ThreeVector Diploe::getField(const ThreeVector& position) const
{
    ThreeVector local_pos = position - m_location;
    double zr = std::abs(local_pos[2]) - 0.5 * m_length;
    if (zr < 0)
    {
        return m_fieldStrength;
    } else
    {
        return m_fieldStrength * getEdge(zr);
    }
}

/* End of Dipole
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/

Quadrupole::Quadrupole(const ThreeVector& location,
    const ThreeVector& fieldStrength, double length, double edgeLength):
FieldBlock(location, fieldStrength,length, edgeLength)
{
}

ThreeVector Quadrupole::getField(const ThreeVector& position) const
{
    ThreeVector local_pos = position - m_location;
    ThreeVector s_pos = ThreeVector(local_pos[1], local_pos[0], local_pos[2]);
    double zr = std::abs(local_pos[2]) - 0.5 * m_length;
    if (zr < 0)
    {
        return m_fieldStrength * s_pos;
    } else
    {
        return m_fieldStrength * s_pos * getEdge(zr);
    }
}