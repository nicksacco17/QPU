
#ifndef MATRIX_H
#define MATRIX_H

#include <complex>
#include <vector>
#include <initializer_list>

using std::vector;
using std::complex;
using std::initializer_list;

class Matrix
{
public:

	Matrix();

	~Matrix();

	Matrix(vector<vector<complex<double>>> in_mat);

	Matrix(vector<complex<double>> in_vec, unsigned int in_row, unsigned int in_col);

	//Matrix(initializer_list<vector<complex<double>>> in_list);

	void print();

	complex<double> get_element(unsigned int row, unsigned int col);

	void set_element(unsigned int row, unsigned int col, complex<double> in_value);

	vector<complex<double>> get_row_order_mat();

	vector<complex<double>> get_col_order_mat();

	unsigned int get_num_rows();

	unsigned int get_num_cols();

	/* ************************************* OPERATORS ************************************** */

	Matrix& operator=(const Matrix& mat);					// Assignment (A = B)

	bool operator==(const Matrix& mat);						// (Strict) Equality (A == B)

	bool operator!=(const Matrix& mat);						// Not Equal (A != B)

	Matrix& operator+=(const Matrix& mat);					// Addition (A += B)

	const Matrix operator+(const Matrix& mat) const;		// Addition (C = A + B)

	Matrix& operator-=(const Matrix& mat);					// Subtraction (A -= B)

	const Matrix operator-(const Matrix& mat) const;		// Subtraction (C = A - B)

	// SCALAR
	Matrix& operator*=(const complex<double> alpha);			// Multiplication (A *= a)

	const Matrix operator*(const complex<double> alpha) const;	// Multiplication (B = A * a)

	Matrix& operator/=(const complex<double> alpha);			// Division (A *= 1/a)

	const Matrix operator/(const complex<double> alpha) const;	// Division (B = A * 1/a)
	
	// MATRIX
	Matrix& operator*=(const Matrix& mat);					// Multiplication (A *= B)

	const Matrix operator*(const Matrix& mat) const;		// Multiplication (C = A * B)

	/* ************************************* OPERATORS ************************************** */

	void transpose();


private:

	unsigned int RC_TO_INDEX(unsigned int row, unsigned int col);

protected:

	unsigned int m_num_row;
	unsigned int m_num_col;

	vector<complex<double>> m_mat;

	complex<double> m_determinant;

	complex<double> m_trace;

};

#endif MATRIX_H