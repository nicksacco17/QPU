
#ifndef MATRIX_H
#define MATRIX_H

#include <complex>
#include <vector>

using std::vector;
using std::complex;

class Matrix
{
public:

	/* **************************** CONSTRUCTORS **************************** */
	Matrix();

	Matrix(unsigned int in_row, unsigned int in_col);

	Matrix(const vector<vector<complex<double>>>& in_mat);

	Matrix(const vector<complex<double>>& in_vec, unsigned int in_row, unsigned int in_col);

	~Matrix();

	/* ********************************************************************** */

	/* ***************************** OPERATORS ****************************** */

	// Assignment (A = B)
	Matrix& operator=(const Matrix& mat);					

	// (Strict) Equality (A == B)
	bool operator==(const Matrix& mat);						

	// Not Equal (A != B)
	bool operator!=(const Matrix& mat);						

	// Addition (A += B)
	Matrix& operator+=(const Matrix& mat);					

	// Addition (C = A + B)
	const Matrix operator+(const Matrix& mat) const;		

	// Subtraction (A -= B)
	Matrix& operator-=(const Matrix& mat);					

	// Subtraction (C = A - B)
	const Matrix operator-(const Matrix& mat) const;		

	// Scalar Multiplication (A *= a)
	Matrix& operator*=(const complex<double> alpha);			

	// Scalar Multiplication (B = A * a)
	const Matrix operator*(const complex<double> alpha) const;	

	// Scalar Division (A *= 1/a)
	Matrix& operator/=(const complex<double> alpha);			

	// Scalar Division (B = A * 1/a)
	const Matrix operator/(const complex<double> alpha) const;	
	
	// Matrix Multiplication (A *= B)
	Matrix& operator*=(const Matrix& mat);					

	// Matrix Multiplication (C = A * B)
	const Matrix operator*(const Matrix& mat) const;		

	/* ********************************************************************** */

	/* *********************** ACCESSORS & MUTATORS ************************* */

	complex<double> get_element(unsigned int row, unsigned int col) const;

	void set_element(unsigned int row, unsigned int col, complex<double> in_value);

	unsigned int get_num_rows() const;

	unsigned int get_num_cols() const;

	void set_dims(unsigned int in_row, unsigned int in_col);

	vector<complex<double>> get_row_order_mat() const;

	vector<complex<double>> get_col_order_mat() const;

	vector<vector<complex<double>>> get_matrix();

	void set_matrix(vector<vector<complex<double>>>& in_mat);

	void set_matrix(vector<complex<double>>& in_vec, unsigned int in_row, unsigned int in_col);

	Matrix get_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2);

	void set_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2, const Matrix& submatrix);

	/* ********************************************************************** */
	
	/* ***************************** FUNCTIONS ****************************** */

	void createIdentityMatrix();

	void transpose();

	void conjugate();

	void hermitian_conjugate();

	void inverse();

	void exponential();

	vector<double> get_eigenvalues();

	double get_eigenvalue(unsigned int index);

	vector<vector<complex<double>>> get_eigenvectors();

	vector<complex<double>> get_eigenvector(unsigned int index);

	complex<double> get_determinant() const;

	complex<double> get_trace() const;

	/* ********************************************************************** */
	
	/* ****************************** UTILITY ******************************* */

	virtual void print() const;

	virtual void print_shape() const;

	void clear();

	void calc_eigenvalues();

	/* ********************************************************************** */

private:

	void i_update_internal();
	void i_calc_trace();
	void i_calc_determinant();

	//vector<<complex<double>> m_eigenvalues;
	
protected:

	//void calc_eigenvalues();

	unsigned int m_num_row;
	unsigned int m_num_col;

	// For simplicity use this field if square matrix (which most/all operators are)
	int m_dim;

	vector<complex<double>> m_mat;
	//vector<vector<complex<double>>> m_eigenvectors;

	vector<complex<double>> m_eigenvectors;
	vector<double> m_eigenvalues;

	complex<double> m_determinant;
	complex<double> m_trace;

	//mutable update_cache m_cache;
};

#endif // MATRIX_H

//struct update_cache
//{
	//bool update_trace_flag;
	//bool update_determinant_flag;
	//bool update_eig_flag;
//};
