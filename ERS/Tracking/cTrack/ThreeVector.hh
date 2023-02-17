#ifndef THREEVECTOR_HH
#define THREEVECTOR_HH

#include <iostream>
#include <cmath>
#include <cassert>


class ThreeVector
{
/**
 * Simple Three vector class, implements basic linear algebra
 * such as dot / cross product 
*/

public:
	// Defult Constructor
	ThreeVector()
	{
		m_data[0] = 0;
		m_data[1] = 0;
		m_data[2] = 0;

	}

	// Copy constructor
	ThreeVector(const ThreeVector &vector)
	{
		m_data[0] = vector.m_data[0];
		m_data[1] = vector.m_data[1];
		m_data[2] = vector.m_data[2];
	}

	// Main init constructor
	ThreeVector(double x, double y, double z)
	{
		m_data[0] = x;
		m_data[1] = y;
		m_data[2] = z;
	}

	~ThreeVector()
	{
	}

	// Prints the vector to the screan
	void Print() const
	{
		std::cout << "[" << m_data[0] << ", " << m_data[1] << ", " 
				  << m_data[2] << "]" << std::endl;
	}

	// Returns the cross product
	ThreeVector Cross(const ThreeVector& vector) const
	{
		ThreeVector newVector;
		newVector.m_data[0] = m_data[1] * vector.m_data[2] 
		                      - m_data[2] * vector.m_data[1];
		newVector.m_data[1] = m_data[2] * vector.m_data[0] 
							  - m_data[0] * vector.m_data[2];
		newVector.m_data[2] = m_data[0] * vector.m_data[1]
		                      - m_data[1] * vector.m_data[0];
		return newVector;
	}

	ThreeVector Norm() const
	{
		return *this / Mag();
	}

	// Returns the dot product
	double Dot(const ThreeVector& vector) const
	{
		return m_data[0] * vector.m_data[0] + m_data[1] * vector.m_data[1]
			   	+ m_data[2] * vector.m_data[2];
	}

	// Returns the magnitude
	double Mag() const
	{
		return std::sqrt(Dot(*this));
	}

	// Returns the magnitude squared
	double Mag2() const
	{
		return Dot(*this);
	}

	// Returns the unit vector pointing from frim this to point
	ThreeVector Direction(const ThreeVector& point) const
	{
		ThreeVector dir;
		dir[0] = point[0] - m_data[0];
		dir[1] = point[1] - m_data[1];
		dir[2] = point[2] - m_data[2];
		dir.Norm();
		return dir;
	}


	// Returns the value of the vector at elementIndex. This method allows 
	// you to edit the vector data.
	double& operator[](unsigned int index)
	{
		assert(index >= 0 && index < 3);
		return m_data[index];
	}

	// Same as above but thish time you can't edit.
	double operator[](unsigned int index) const
	{
		assert(index >= 0 && index < 3);
		return m_data[index];
	}

	// Copy vecotor to new vector
	ThreeVector& operator=(const ThreeVector &vector)
	{

		// Self assigment gaurd
		if (this == &vector)
		{
			return *this;
		}
		m_data[0] = vector.m_data[0];
		m_data[1] = vector.m_data[1];
		m_data[2] = vector.m_data[2];

		return *this;
	}

	//vector-vector operations:
	friend ThreeVector operator+(const ThreeVector &vector1,
		const ThreeVector &vector2)
	{
		ThreeVector newVector;
		newVector.m_data[0] = vector1.m_data[0] + vector2.m_data[0];
		newVector.m_data[1] = vector1.m_data[1] + vector2.m_data[1];
		newVector.m_data[2] = vector1.m_data[2] + vector2.m_data[2];
		return newVector;
	};

	friend ThreeVector operator-(const ThreeVector &vector1,
		const ThreeVector &vector2)
	{
		ThreeVector newVector;
		newVector.m_data[0] = vector1.m_data[0] - vector2.m_data[0];
		newVector.m_data[1] = vector1.m_data[1] - vector2.m_data[1];
		newVector.m_data[2] = vector1.m_data[2] - vector2.m_data[2];
		return newVector;
	};

	friend ThreeVector operator*(const ThreeVector &vector1,
		const ThreeVector &vector2)
	{
		ThreeVector newVector;
		newVector.m_data[0] = vector1.m_data[0] * vector2.m_data[0];
		newVector.m_data[1] = vector1.m_data[1] * vector2.m_data[1];
		newVector.m_data[2] = vector1.m_data[2] * vector2.m_data[2];
		return newVector;
	};

	friend ThreeVector operator/(const ThreeVector &vector1, 
		const ThreeVector &vector2)
	{
		ThreeVector newVector;
		newVector.m_data[0] = vector1.m_data[0] / vector2.m_data[0];
		newVector.m_data[1] = vector1.m_data[1] / vector2.m_data[1];
		newVector.m_data[2] = vector1.m_data[2] / vector2.m_data[2];
		return newVector;
	};

	//T-vector operations:
	friend ThreeVector operator+(double scalar, const ThreeVector &vector)
	{
		ThreeVector newVector;
		newVector.m_data[0] = scalar + vector.m_data[0];
		newVector.m_data[1] = scalar + vector.m_data[1];
		newVector.m_data[2] = scalar + vector.m_data[2];
		return newVector;
	};

	friend ThreeVector operator+(const ThreeVector &vector, double scalar)
	{
		ThreeVector newVector;
		newVector.m_data[0] = vector.m_data[0] + scalar;
		newVector.m_data[1] = vector.m_data[1] + scalar;
		newVector.m_data[2] = vector.m_data[2] + scalar;
		return newVector;
	};

	friend ThreeVector operator-(double scalar, const ThreeVector &vector)
	{
		ThreeVector newVector;
		newVector.m_data[0] = scalar - vector.m_data[0];
		newVector.m_data[1] = scalar - vector.m_data[1];
		newVector.m_data[2] = scalar - vector.m_data[2];
		return newVector;
	};

	friend ThreeVector operator-(const ThreeVector &vector, double scalar)
	{
		ThreeVector newVector;
		newVector.m_data[0] = vector.m_data[0] - scalar;
		newVector.m_data[1] = vector.m_data[1] - scalar;
		newVector.m_data[2] = vector.m_data[2] - scalar;
		return newVector;
	};

	friend ThreeVector operator*(double scalar, const ThreeVector &vector)
	{
		ThreeVector newVector;
		newVector.m_data[0] = scalar * vector.m_data[0];
		newVector.m_data[1] = scalar * vector.m_data[1];
		newVector.m_data[2] = scalar * vector.m_data[2];
		return newVector;
	};

	friend ThreeVector operator*(const ThreeVector &vector, double scalar)
	{
		ThreeVector newVector;
		newVector.m_data[0] = vector.m_data[0] * scalar;
		newVector.m_data[1] = vector.m_data[1] * scalar;
		newVector.m_data[2] = vector.m_data[2] * scalar;
		return newVector;
	};

	friend ThreeVector operator/(double scalar, const ThreeVector &vector)
	{
		ThreeVector newVector;
		newVector.m_data[0] = scalar / vector.m_data[0];
		newVector.m_data[1] = scalar / vector.m_data[1];
		newVector.m_data[2] = scalar / vector.m_data[2];
		return newVector;
	};

	friend ThreeVector operator/(const ThreeVector &vector, double scalar)
	{
		ThreeVector newVector;
		newVector.m_data[0] = vector.m_data[0] / scalar;
		newVector.m_data[1] = vector.m_data[1] / scalar;
		newVector.m_data[2] = vector.m_data[2] / scalar;
		return newVector;
	};


private:
	double m_data[3];
};

#endif