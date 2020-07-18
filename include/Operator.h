#pragma once

#include <complex>
#include <vector>
#include <string>
#include <initializer_list>

#include "Utility.h"

using std::string;
using std::complex;
using std::vector;
using std::initializer_list;

class Operator
{
public:

	/* ************************************ CONSTRUCTORS ************************************ */
	
	Operator();

	Operator(unsigned int in_size_x, unsigned int in_size_y);

	Operator(const vector<vector<complex<double>>>& in_matrix);

	~Operator();

	/* ************************************ CONSTRUCTORS ************************************ */

	/* ************************************* OPERATORS ************************************** */

	Operator& operator=(const Operator& mat);					// Assignment (A = B)

	bool operator==(const Operator& mat);						// (Strict) Equality (A == B)

	bool operator!=(const Operator& mat);						// Not Equal (A != B)

	Operator& operator+=(const Operator& mat);					// Addition (A += B)

	const Operator operator+(const Operator& mat) const;		// Addition (C = A + B)

	Operator& operator-=(const Operator& mat);					// Subtraction (A -= B)

	const Operator operator-(const Operator& mat) const;		// Subtraction (C = A - B)

	Operator& operator*=(const complex<double> a);				// Multiplication (A *= a)

	Operator& operator*=(const Operator& mat);					// Multiplication (A *= B)

	const Operator operator*(const complex<double> a) const;	// Multiplication (B = A * a)

	const Operator operator*(const Operator& mat) const;		// Multiplication (C = A * B)

	/* ************************************* OPERATORS ************************************** */

	/* ************************************* FUNCTIONS ************************************** */

	// Getters

	unsigned int get_dim_x() const;

	unsigned int get_dim_y() const;

	complex<double> get_trace() const;

	complex<double> get_determinant() const;

	complex<double> get_element(int row, int col) const
	{
		return m_matrix[row][col];
	}

	void set_element(int row, int col, complex<double> val);

	vector<vector<complex<double>>> get_matrix() const;

	// Populate the contents of the state with an array of elements
	void populate(const vector<vector<complex<double>>>& in_matrix);

	void createIdentityMatrix();

	virtual void print() const;

	void print_shape() const;

	// ---------------------------------------------------
	// Special Matrix Forms

	// Return true if the matrix is a square (n x n) matrix
	bool is_Square();

	// Return true if the matris is singular (i.e. Non-invertible)
	// Requires square matrix
	bool is_Singular();

	// Return true if the matrix is a Hermitian matrix (A = A*T)
	// Requires square matrix
	bool is_Hermitian();
	
	// Return true if the matrix is a Unitary matrix (AA*T = A*TA = I)
	// Requires square matrix
	bool is_Unitary();

	// Return true if the matrix is an Orthogonal matrix (AT = A-1)
	// Requires square matrix
	bool is_Orthogonal();

	// Return true if the matrix is a Normal matrix (A*TA == AA*T)
	bool is_Normal();

	void inverse();

	void transpose();

	void conjugate();

	void hermitian_congugate();

	void clear()
	{
		m_matrix = vector<vector<complex<double>>>(m_dim_x, vector<complex<double>>(m_dim_y, 0.0));
	}

	vector<vector<complex<double>>>::const_iterator get_start_address() const
	{
		return m_matrix.begin();
	}

	vector<vector<complex<double>>>::const_iterator get_end_address() const
	{
		return m_matrix.end();
	}

	Operator get_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2);

	void set_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2, const vector<vector<complex<double>>>& submatrix);

	void set_dim(unsigned int in_dim_x, unsigned int in_dim_y)
	{
		m_dim_x = in_dim_x;
		m_dim_y = in_dim_y;
		m_matrix = vector<vector<complex<double>>>(m_dim_x, vector<complex<double>>(m_dim_y, 0.0));
	}

	void augment(Operator& A);


	/* ************************************* FUNCTIONS ************************************** */

protected:

	unsigned int m_dim_x;
	unsigned int m_dim_y;

	complex<double> m_trace;
	complex<double> m_determinant;
	vector<vector<complex<double>>> m_matrix;
	
private:

	void i_calc_determinant();
	void i_calc_trace();

};

