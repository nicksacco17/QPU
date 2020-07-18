#pragma once

#include "State.h"
#include "Operator.h"
#include <complex>

using std::complex;

// These definitely should be in this library because they rely on two different objects

void inner_product(complex<double>& result, const State& psi_1, const State& psi_2);

void inner_product_w_operator(complex<double>& result, const State& psi_1, const Operator& A, const State& psi_2);

void outer_product(Operator& mat, const State& psi_1, const State& psi_2);

//complex<double> inner_product_w_operator(const State& psi_1, const State& psi_2, const Operator& A);

vector<complex<double>> eigval_mat2(Operator& mat);

vector<complex<double>> eigval_mat3(Operator& mat);




State op_rhs(const Operator& A, const State& psi);

State tensor_product(State& psi_1, State& psi_2);

// Maybe all of these can be in the operator class because they act on one operator only?

Operator complex_conjugate(Operator& A);

Operator transpose(Operator& A);

Operator hermitian_conjugate(Operator& A);

bool is_orthogonal(const State& psi_1, const State& psi_2);

// Decompose the current matrix (A) into two matrices (Q, R) according to QR decomposition algorithm
// A = QR SUCH THAT Q is an orthogonal matrix (Q^T = INV(Q)), R is an upper triangular matrix
// A = QR SUCH THAT Q is a unitary matrix (Q* = INV(Q)), R is an upper triangular matrix
void QR_decomposition(Operator& A, Operator& Q, Operator& R);


// Complex
//void Givens_rotation(unsigned int dim_x, unsigned int dim_y, unsigned int row, unsigned int col, complex<double> a, complex<double> b);

struct Givens_rotation_t
{
	complex<double> c;
	complex<double> s;
	double radius;
};

double diagonal_error(const Operator& A, const Operator& B);

double approx_diagonal(const Operator& A, const Operator& B);

double sub_diagonal_error(const Operator& A, const Operator& B);

void Givens_coefficient(const complex<double>& a, const complex<double>& b, Givens_rotation_t& Gij);

void QR_Algorithm(Operator& A, Operator& A_decomposition, vector<complex<double>>& A_eigenvalues);

void Givens_rotation_matrix(const unsigned int row, const unsigned int col, const complex<double>& a, const complex<double>& b, Operator& G);

void Hessenberg_reduction(Operator& H);

void Strassen_matrix_multiplication(Operator& C, Operator& A, Operator& B);

// Pure real - return 2D matrix of doubles
vector<vector<complex<double>>> Givens_rotation(unsigned int dim_x, unsigned int dim_y, unsigned int row, unsigned int col, complex<double> a, complex<double> b);

void tensor_product(State& result, const State& psi_1, const State& psi_2);

// A ** B
void tensor_product(Operator& result, const Operator& A, const Operator& B);

/*const Operator BELL_TO_STANDARD_BASIS({	{ INV_SQRT2, INV_SQRT2, 0, 0 },
										{ 0, 0, INV_SQRT2, INV_SQRT2 },
										{ 0, 0, INV_SQRT2, -1.0 * INV_SQRT2 },
										{ INV_SQRT2, -1.0 * INV_SQRT2, 0, 0 } });

const Operator STANDARD_TO_BELL_BASIS({	{ INV_SQRT2, 0, 0, INV_SQRT2 },
										{ INV_SQRT2, 0, 0, -1.0 * INV_SQRT2 },
										{ 0, INV_SQRT2, INV_SQRT2, 0 },
										{ 0, INV_SQRT2, -1.0 * INV_SQRT2, 0 } });
										*/