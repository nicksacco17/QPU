
#include "../include/Matrix.h"
#include "../include/Utility.h"

#include <iostream>
#include <algorithm>
#include <functional>

using std::cout;
using std::endl;

Matrix::Matrix()
{
	m_num_row = 2;
	m_num_col = 2;

	m_mat = { 1, 0, 1, 0 };

	m_determinant = 1;
	m_trace = 2;
}

Matrix::~Matrix()
{
	m_mat.clear();
}

Matrix::Matrix(vector<vector<complex<double>>> in_mat)
{
	m_num_row = in_mat.size();
	m_num_col = in_mat.at(0).size();

	for (unsigned int i = 0; i < in_mat.size(); i++)
	{
		m_mat.insert(m_mat.end(), in_mat.at(i).begin(), in_mat.at(i).end());
	}

	m_determinant = 9999;
	m_trace = 9999;
}

Matrix::Matrix(vector<complex<double>> in_vec, unsigned int in_row, unsigned int in_col)
{
	m_mat = in_vec;

	m_num_row = in_row;
	m_num_col = in_col;

	m_determinant = 9999;
	m_trace = 9999;
}

void Matrix::print()
{
	cout << "---------- PRINT MATRIX ----------" << endl;

	cout << "DIMENSION: (" << m_num_row << " x " << m_num_col << ")" << endl;
	cout << "TRACE: " << m_trace << endl;
	cout << "DETERMINANT: " << m_determinant << endl;

	cout << "ELEMENTS:\n" << endl;

	for (unsigned int i = 0; i < m_num_row; i++)
	{
		cout << "| ";
		for (unsigned int j = 0; j < m_num_col; j++)
		{
			cout << m_mat.at(RC_TO_INDEX(i, j)) << " ";
		}
		cout << "|" << endl;
	}

	cout << "\n---------- PRINT MATRIX ----------" << endl;
}

unsigned int Matrix::get_num_rows()
{
	return m_num_row;
}

unsigned int Matrix::get_num_cols()
{
	return m_num_col;
}

unsigned int Matrix::RC_TO_INDEX(unsigned int row, unsigned int col)
{
	return ((row * m_num_col) + col);
}

complex<double> Matrix::get_element(unsigned int row, unsigned int col)
{
	return RC_TO_INDEX(row, col);
}

void Matrix::set_element(unsigned int row, unsigned int col, complex<double> in_value)
{
	m_mat.at(RC_TO_INDEX(row, col)) = in_value;
}

vector<complex<double>> Matrix::get_row_order_mat()
{
	return m_mat;
}

vector<complex<double>> Matrix::get_col_order_mat()
{
	this->transpose();

	vector<complex<double>> COL_MAT = this->m_mat;

	this->transpose();

	return COL_MAT;
}

/* ************************************* OPERATORS ************************************** */

// Assignment (A = B)
Matrix& Matrix::operator=(const Matrix& mat)					
{
	if (this != &mat)
	{
		this->m_num_row = mat.m_num_row;
		this->m_num_col = mat.m_num_col;
		this->m_mat = mat.m_mat;

		this->m_determinant = mat.m_determinant;
		this->m_trace = mat.m_trace;
	}

	return *this;
}

// (Strict) Equality (A == B)
bool Matrix::operator==(const Matrix& mat)					
{
	if (this->m_num_row == mat.m_num_row && this->m_num_col == mat.m_num_col)
	{
		for (int i = 0; i < this->m_mat.size(); i++)
		{
			if (this->m_mat.at(i) != mat.m_mat.at(i))
			{
				return false;
			}
		}
	}
	else
	{
		return false;
	}

	return true;
}

// Not Equal (A != B)
bool Matrix::operator!=(const Matrix& mat)						
{
	return !(*this == mat);
}

// SCALAR

// Addition (A += B)
Matrix& Matrix::operator+=(const Matrix& mat)					
{
	std::transform(this->m_mat.begin(), this->m_mat.end(), mat.m_mat.begin(), this->m_mat.begin(), std::plus<complex<double>>());
}

// Addition (C = A + B)
const Matrix Matrix::operator+(const Matrix& mat) const		
{
	Matrix TEMP = *this;
	TEMP += mat;

	return TEMP;
}

// Subtraction (A -= B)
Matrix& Matrix::operator-=(const Matrix& mat)					
{
	std::transform(this->m_mat.begin(), this->m_mat.end(), mat.m_mat.begin(), this->m_mat.begin(), std::minus<complex<double>>());
}

// Subtraction (C = A - B)
const Matrix Matrix::operator-(const Matrix& mat) const		
{
	Matrix TEMP = *this;
	TEMP -= mat;

	return TEMP;
}

// Multiplication (A *= a)
Matrix& Matrix::operator*=(const complex<double> alpha)			
{
	std::transform(this->m_mat.begin(), this->m_mat.end(), this->m_mat.begin(), std::bind1st(std::multiplies<complex<double>>(), alpha));
}

// Multiplication (B = A * a)
const Matrix Matrix::operator*(const complex<double> alpha) const	
{
	Matrix TEMP = *this;
	TEMP *= alpha;

	return TEMP;
}

// Division (A *= 1/a)
Matrix& Matrix::operator/=(const complex<double> alpha)
{
	if (std::real(alpha) > 1e-10 && std::imag(alpha) > 1e-10)
	{
		std::transform(this->m_mat.begin(), this->m_mat.end(), this->m_mat.begin(), std::bind1st(std::multiplies<complex<double>>(), (1.0 / alpha)));
	}
}

// Division (B = A * 1/a)
const Matrix Matrix::operator/(const complex<double> a) const
{
	Matrix TEMP = *this;
	TEMP /= a;

	return TEMP;
}

// MATRIX

// Multiplication (A *= B)
Matrix& Matrix::operator*=(const Matrix& mat)					
{
	if (this != &mat && this->m_num_col == mat.m_num_row)
	{
		unsigned int NUM_ELEMENTS = this->m_num_row * mat.m_num_col;
		unsigned int COMMON_DIM = this->m_num_col;
		vector<complex<double>> NEW_MAT(NUM_ELEMENTS, 0.0);

		for (unsigned int i = 0; i < this->m_num_row; i++)
		{
			for (unsigned int k = 0; k < COMMON_DIM; k++)
			{
				for (unsigned int j = 0; j < mat.m_num_col; j++)
				{
					NEW_MAT.at(RC_TO_INDEX(i, j)) += (this->m_mat.at(RC_TO_INDEX(i, k)) * mat.m_mat.at(RC_TO_INDEX(k, j)));
				}
			}
		}

		this->m_mat = NEW_MAT;
		this->m_num_row = this->m_num_row;
		this->m_num_col = mat.m_num_col;

		// Update trace
		// Update determinant
	}

	return *this;
}

// Multiplication (C = A * B)
const Matrix Matrix::operator*(const Matrix& mat) const		
{
	Matrix TEMP = *this;
	TEMP *= mat;

	return TEMP;
}

/* ************************************* OPERATORS ************************************** */

void Matrix::transpose()
{
	unsigned int new_num_row = this->m_num_col;
	unsigned int new_num_col = this->m_num_row;

	complex<double> TEMP;

	// If square matrix
	if (m_num_row == m_num_col)
	{
		for (unsigned int row = 0; row < m_num_row - 1; row++)
		{
			for (unsigned int col = row + 1; col < m_num_col; col++)
			{
				TEMP = m_mat.at(RC_TO_INDEX(row, col));
				m_mat.at(RC_TO_INDEX(row, col)) = m_mat.at(RC_TO_INDEX(col, row));
				m_mat.at(RC_TO_INDEX(col, row)) = TEMP;
			}
		}
	}

	// Else if not square
	else
	{
		vector<complex<double>> NEW_MAT(new_num_row * new_num_col, 0.0);

		for (unsigned int row = 0; row < m_num_row; row++)
		{
			for (unsigned int col = 0; col < m_num_col; col++)
			{
				NEW_MAT.at(RC_TO_INDEX(col, row)) = m_mat.at(RC_TO_INDEX(row, col));
			}
		}
		m_mat = NEW_MAT;
	}

	m_num_row = new_num_row;
	m_num_col = new_num_col;

	// Update trace
	// Update determinant
	
}	
