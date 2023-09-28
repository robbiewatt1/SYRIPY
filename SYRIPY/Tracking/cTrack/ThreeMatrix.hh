#ifndef THREEMATRIX_HH
#define THREEMATRIX_HH

class ThreeMatrix
{
private:
	double m_data[9];
public:

	ThreeMatrix()
	{
		Zeros();
	}

	ThreeMatrix(const ThreeMatrix &matrix)
	{
		for (int i = 0; i < 9; i++)
		{
			m_data[i] = matrix.m_data[i];
		}
	}

	~ThreeMatrix()
	{
	}

	void Zeros()
	{
		for (int i = 0; i < 9; i++)
		{
			m_data[i] = 0.0;
		}
	}

	void Identity()
	{
		m_data[0] = 1.0;
		m_data[1] = 0.0;
		m_data[2] = 0.0;
		m_data[3] = 0.0;
		m_data[4] = 1.0;
		m_data[5] = 0.0;
		m_data[6] = 0.0;
		m_data[7] = 0.0;
		m_data[8] = 1.0;
	}

	void Print() const
	{
		std::cout << "[";
		for (unsigned int i = 0; i < 3; i++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				std::cout << m_data[3*i+j] << ", ";
			}
			if (i != 2)
			{
				std::cout << "\n ";
			} else
			{
				std::cout << "]\n";
			}	
		}
	}

	double Determinant() const
	{
		return  m_data[0] * (m_data[4] * m_data[8] - m_data[5] * m_data[7])
			  - m_data[1] * (m_data[3] * m_data[8] - m_data[5] * m_data[6])
			  + m_data[2] * (m_data[3] * m_data[7] - m_data[4] * m_data[6]);
	}

	ThreeMatrix Inverse() const
	{
		ThreeMatrix inverse;
		inverse.m_data[0] = m_data[4] * m_data[8] - m_data[5] * m_data[7];
		inverse.m_data[1] = m_data[2] * m_data[7] - m_data[1] * m_data[8]; 
		inverse.m_data[2] = m_data[1] * m_data[5] - m_data[2] * m_data[4];
		inverse.m_data[3] = m_data[5] * m_data[6] - m_data[3] * m_data[8];
		inverse.m_data[4] = m_data[0] * m_data[8] - m_data[2] * m_data[6];
		inverse.m_data[5] = m_data[2] * m_data[3] - m_data[0] * m_data[5];
		inverse.m_data[6] = m_data[3] * m_data[7] - m_data[4] * m_data[6];
		inverse.m_data[7] = m_data[1] * m_data[6] - m_data[0] * m_data[7];
		inverse.m_data[8] = m_data[0] * m_data[4] - m_data[1] * m_data[3];
		return inverse * (1.0 / Determinant());
	}

	double* operator[](unsigned int index)
	{
		assert(index >= 0 && index < 3);
		return &m_data[3*index];
	}

	const double* operator[](unsigned int index) const
	{
		assert(index >= 0 && index < 3);
		return &m_data[3*index];
	}

	friend ThreeMatrix operator+(const ThreeMatrix &matrix1, const ThreeMatrix &matrix2)
	{
		ThreeMatrix newMatrix;
		for (int i = 0; i < 9; i++)
		{
			newMatrix.m_data[i] = matrix1.m_data[i] + matrix2.m_data[i];
		}
		return newMatrix;
	}

	friend ThreeMatrix operator*(const ThreeMatrix &matrix1, const ThreeMatrix &matrix2)
	{
		ThreeMatrix newMatrix;
		for (unsigned int i = 0; i < 3; i++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				for (unsigned int k = 0; k < 3; k++)
				{
					newMatrix.m_data[3*i+j] += matrix1.m_data[3*i+k] * matrix2.m_data[3*k+j];
				}
			}
		}
		return newMatrix;
	}

	friend ThreeMatrix operator*(const ThreeMatrix &matrix, double scalar)
	{
		ThreeMatrix newMatrix;
		for (int i = 0; i < 9; i++)
		{
			newMatrix.m_data[i] = matrix.m_data[i] * scalar;
		}
		return newMatrix;
	}
};
#endif