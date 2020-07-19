
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

	Operator(string mat_type, unsigned int in_dim);

	~Operator();

	/* ********************************************************************** */

	/* ****************************** UTILITY ******************************* */

	void createIdentityMatrix();

	void print() const;

	void print_shape() const;

	/* ********************************************************************** */

protected:

	vector<double> m_eigenvalues;

private:

	void i_calc_determinant();
	void i_calc_trace();

};

#endif // OPERATOR_H
