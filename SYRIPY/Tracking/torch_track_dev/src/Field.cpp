#include "Field.hh"

FieldBlock::FieldBlock(const vectorType& location, 
    const vectorType& fieldStrength, scalarType length, scalarType edgeLength):
m_location(location), m_fieldStrength(fieldStrength), m_length(length), 
    m_edgeLength(edgeLength)
{
    m_edgeScaleFact = m_edgeLength / 1.23789045853;
    m_constLengthHalf = 0.5 * (length - 1.2689299897 * m_edgeLength);

}

scalarType FieldBlock::getEdge(scalarType z) const
{
    scalarType zp2 = z * z / (m_edgeScaleFact * m_edgeScaleFact);
    return 1. / ((1. + zp2) * (1. + zp2));
}

/* End of FieldBlock
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/

void FieldContainer::addElement(int order, const vectorType& location,
        const vectorType& fieldStrength, scalarType length,
        scalarType edgeLength)
{
    //FieldBlock element;
    if (order == 1) // dipole
    {
        Dipole* element = new Dipole(location, fieldStrength, length,
            edgeLength);
        m_fieldContainer.push_back(element);
    } else if (order == 2)
    {
        Quadrupole* element = new Quadrupole(location, fieldStrength, length,
            edgeLength);
        m_fieldContainer.push_back(element);
    } else
    {
        std::cerr << "Error: Field order must be 1 for dipole or 2 for " 
                     "quadrupole. Instead received: " << order << std::endl;
        return;
    }
    
}

vectorType FieldContainer::getField(const vectorType& position) const
{
    vectorType field = m_fieldContainer[0]->getField(position);
    for (long unsigned int i = 1; i < m_fieldContainer.size(); i++)
    {
        field = field + m_fieldContainer[i]->getField(position);
    }
    return field;
}

/* End of FieldContainer
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/

Dipole::Dipole(const vectorType& location, const vectorType& fieldStrength,
        scalarType length, scalarType edgeLength):
FieldBlock(location, fieldStrength,length, edgeLength)
{
}

vectorType Dipole::getField(const vectorType& position) const
{
    vectorType local_pos = position - m_location;
    scalarType zr = std::abs(local_pos.getItem(2)) - m_constLengthHalf;
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


Quadrupole::Quadrupole(const vectorType& location,
    const vectorType& fieldStrength, scalarType length, scalarType edgeLength):
FieldBlock(location, fieldStrength,length, edgeLength)
{
}

vectorType Quadrupole::getField(const vectorType& position) const
{
    vectorType local_pos = position - m_location;
    vectorType s_pos = vectorType({local_pos.getItem(1), local_pos.getItem(0),
        local_pos.getItem(2)});
    
    scalarType zr = std::abs(local_pos.getItem(2)) - 0.5 * m_length;
    if (zr < 0.)
    {
        return m_fieldStrength * s_pos;
    } else
    {
        return m_fieldStrength * s_pos * getEdge(zr);
    }
}