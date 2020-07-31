
#ifndef OPERATOR_H
#define OPERATOR_H

#include <complex>
#include <vector>
#include <string>

#include "Utility.h"
#include "Matrix.h"

using std::string;
using std::complex;
using std::vector;

class Operator : public Matrix
{
public:

	/* **************************** CONSTRUCTORS **************************** */
	
	Operator();

	Operator(unsigned int in_row, unsigned int in_col);

	Operator(const vector<vector<complex<double>>>& in_mat);

	Operator(const vector<complex<double>>& in_vec, unsigned int in_row, unsigned int in_col);

	Operator(string mat_type, unsigned int in_dim, double lower_range, double upper_range, long unsigned int seed);

	~Operator();

	Operator& operator=(const Matrix& mat);

	Operator get_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2);

	/* ********************************************************************** */

	/* *********************** ACCESSORS & MUTATORS ************************* */

	vector<double> get_eigenvalues();

	double get_eigenvalue(unsigned int index);

	vector<vector<complex<double>>> get_eigenvectors();

	vector<complex<double>> get_eigenvector(unsigned int index);

	/* ********************************************************************** */

	/* ***************************** FUNCTIONS ****************************** */

	void inverse();

	void exponential();

	void calc_eigens();

	/* ********************************************************************** */

	/* ****************************** UTILITY ******************************* */

	void print() const;

	void print_shape() const;

	/* ********************************************************************** */

protected:

	vector<double> m_eigenvalues;
	vector<vector<complex<double>>> m_eigenvectors;

private:

	vector<complex<double>> m_eigenvectors_UNFORMATTED;

	void i_calc_determinant();
	void i_calc_trace();

};

#endif // OPERATOR_H
