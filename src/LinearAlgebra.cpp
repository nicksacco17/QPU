
#include "../include/LinearAlgebra.h"
#include "../include/Utility.h"
#include <assert.h>

#include <iostream>
#include <numeric>
#include <ctime>
#include <functional>
#include <algorithm>
#include <chrono>

#include <math.h>

using std::cout;
using std::endl;
using std::vector;
using std::complex;

void inner_product(complex<double>& result, const State& psi_l, const State& psi_r)
{

#ifdef USE_GPU
	
	cout << "GPU INNER PRODUCT..." << endl;

#else

	cout << "CPU INNER PRODUCT..." << endl;

	if (psi_l.get_dim() == psi_r.get_dim())
	{
		unsigned int DIM = psi_l.get_dim();
		complex<double> l_inner_product = 0.0;

		for (unsigned int i = 0; i < DIM; i++)
		{
			l_inner_product += (std::conj(psi_l.get_element(i)) * psi_r.get_element(i));
		}

		result = l_inner_product;
	}

#endif

	return;
}

void inner_product_w_operator(complex<double>& result, const State& psi_l, const Operator& A, const State& psi_r)
{

#ifdef USE_GPU
	
	cout << "GPU INNER PRODUCT W/OPERATOR..." << endl;

#else

	cout << "CPU INNER PRODUCT W/OPERATOR..." << endl;

#endif

	return;
}

void outer_product(Operator& result, const State& psi_l, const State& psi_r)
{

#ifdef USE_GPU
	
	cout << "GPU OUTER PRODUCT..." << endl;

#else
	
	cout << "CPU OUTER PRODUCT..." << endl;

	unsigned int L_DIM = psi_l.get_dim();
	unsigned int R_DIM = psi_r.get_dim();

	Operator TEMP(L_DIM, R_DIM);

	for (unsigned int i = 0; i < L_DIM; i++)
	{
		for (unsigned int j = 0; j < R_DIM; j++)
		{
			TEMP.set_element(i, j, psi_l.get_element(i) * std::conj(psi_r.get_element(j)));
		}
	}
	result = TEMP;

#endif

	return;
}

void tensor_product(State& result, const State& psi_1, const State& psi_2)
{

#ifdef USE_GPU
	
	cout << "GPU STATE TENSOR PRODUCT..." << endl;

#else
	
	cout << "CPU STATE TENSOR PRODUCT..." << endl;

	unsigned int DIM_PSI1 = psi_1.get_dim();
	unsigned int DIM_PSI2 = psi_2.get_dim();
	unsigned int TENSOR_DIM = DIM_PSI1 * DIM_PSI2;

	vector<complex<double>> tensor_elements(TENSOR_DIM, 0.0);

	for (unsigned int i = 0; i < DIM_PSI2; i++)
	{
		std::transform(psi_2.get_start_address(), psi_2.get_end_address(), tensor_elements.begin() + (DIM_PSI2 * i), std::bind1st(std::multiplies<complex<double>>(), psi_1.get_element(i)));
	}

	result.populate(tensor_elements);

#endif

	return;
}

void tensor_product(Operator& result, const Operator& A, const Operator& B)
{

#ifdef USE_GPU
	
	cout << "GPU OPERATOR TENSOR PRODUCT..." << endl;

#else
	
	//cout << "CPU OPERATOR TENSOR PRODUCT..." << endl;

	unsigned int A_NUM_ROW = A.get_num_rows();
	unsigned int A_NUM_COL = A.get_num_cols();
	unsigned int B_NUM_ROW = B.get_num_rows();
	unsigned int B_NUM_COL = B.get_num_cols();

	unsigned int row_stride = B_NUM_ROW - 1;
	unsigned int col_stride = B_NUM_COL - 1;

	unsigned int T_NUM_ROW = A_NUM_ROW * B_NUM_ROW;
	unsigned int T_NUM_COL = A_NUM_COL * B_NUM_COL;

	//unsigned int row_start = 0;
	//unsigned int col_start = 0;

	result.set_dims(T_NUM_ROW, T_NUM_COL);

	for (unsigned int i = 0; i < A_NUM_ROW; i++)
	{
		//cout << "[ROW] = [" << i << "]" << endl;
		//row_start = A_DIM_Y * i;
		for (unsigned int j = 0; j < A_NUM_COL; j++)
		{
			//col_start = A_DIM_X * j;

			result.set_submatrix(B_NUM_COL * i, (B_NUM_COL * i) + col_stride, B_NUM_ROW * j, (B_NUM_ROW * j) + row_stride, (B * A.get_element(i, j)));
			//result.set_submatrix(row_start, row_start + v_stride, col_start, col_start + h_stride, (A * B.get_element(i, j)).get_matrix());
		}
	}

#endif

	return;
}

void op_RHS(State& result, const Operator& A, const State& psi)
{

#ifdef USE_GPU
	
	cout << "GPU OP RHS PRODUCT..." << endl;

#else
	
	cout << "CPU OP RHS PRODUCT..." << endl;

#endif

	return;
}

#ifdef HAHAHAHA
#define QR_ALG_ERROR 1e-20
#define QR_ALG_MAX_IT 10000

void inner_product(complex<double>& result, const State& psi_1, const State& psi_2)
{
	complex<double> l_inner_product = 0.0;
	complex<double> INIT = 0.0;

	l_inner_product = std::inner_product(psi_1.get_start_address(), psi_1.get_end_address(), psi_2.get_start_address(), INIT);

	result = l_inner_product;
}

void inner_product_w_operator(complex<double>& result, const State& psi_1, const Operator& A, const State& psi_2)
{

}

// NOPE THIS IS NOT WORKING?
void tensor_product(State& result, const State& psi_1, const State& psi_2)
{
	unsigned int DIM_PSI1 = psi_1.get_dim();
	unsigned int DIM_PSI2 = psi_2.get_dim();
	unsigned int tensor_dim = DIM_PSI1 * DIM_PSI2;

	vector<complex<double>> tensor_elements(tensor_dim, 0.0);

	for (int i = 0; i < DIM_PSI2; i++)
	{
		std::transform(psi_2.get_start_address(), psi_2.get_end_address(), tensor_elements.begin() + (DIM_PSI2 * i), std::bind1st(std::multiplies<complex<double>>(), psi_1.get_element(i)));
		//std::transform(psi_1.get_start_address(), psi_1.get_end_address(), tensor_elements.begin() + (DIM_PSI1 * i), std::bind1st(std::multiplies<complex<double>>(), psi_2.get_element(i)));
	}

	result.populate(tensor_elements);
}

// A ** B

// MAY HAVE FIXED THIS NO IDEA
void tensor_product(Operator& result, const Operator& A, const Operator& B)
{
	unsigned int A_DIM_X = A.get_dim_x();
	unsigned int A_DIM_Y = A.get_dim_y();
	unsigned int B_DIM_X = B.get_dim_x();
	unsigned int B_DIM_Y = B.get_dim_y();

	unsigned int h_stride = B_DIM_X - 1;
	unsigned int v_stride = B_DIM_Y - 1;

	unsigned int tensor_dim_x = A_DIM_X * B_DIM_X;
	unsigned int tensor_dim_y = A_DIM_Y * B_DIM_Y;

	//unsigned int row_start = 0;
	//unsigned int col_start = 0;

	result.set_dim(tensor_dim_x, tensor_dim_y);

	for (int i = 0; i < A_DIM_X; i++)
	{
		//cout << "[ROW] = [" << i << "]" << endl;
		//row_start = A_DIM_Y * i;
		for (int j = 0; j < A_DIM_Y; j++)
		{
			//col_start = A_DIM_X * j;

			result.set_submatrix(B_DIM_Y * i, (B_DIM_Y * i) + v_stride, B_DIM_X * j, (B_DIM_X * j) + h_stride, (B * A.get_element(i, j)).get_matrix());
			//result.set_submatrix(row_start, row_start + v_stride, col_start, col_start + h_stride, (A * B.get_element(i, j)).get_matrix());
		}
	}
}

void outer_product(Operator& mat, const State& psi_1, const State& psi_2)
{
	//vector<complex<double>> psi2_bra = psi_2.get_vector();
	vector<complex<double>> new_row;// (psi_2.get_dim(), 0.0);
	vector<vector<complex<double>>> new_op;

	//new_row.reserve(psi_2.get_dim());
	//new_op.reserve(psi_1.get_dim() * psi_2.get_dim());

	//for (int i = 0; i < psi2_bra.size(); i++)
	//{
		//psi2_bra[i] = std::conj(psi2_bra[i]);
	//}

	//int j = 0;
	for (vector<complex<double>>::const_iterator psi1_it = psi_1.get_start_address(); psi1_it != psi_1.get_end_address(); psi1_it++)
	{
		for (vector<complex<double>>::const_iterator psi2_it = psi_2.get_start_address(); psi2_it != psi_2.get_end_address(); psi2_it++)
		{
			//new_row[j] = ((*psi1_it) * std::conj(*psi2_it));
			new_row.push_back(((*psi1_it) * std::conj(*psi2_it)));
			//j++;
		}
		new_op.push_back(new_row);
		new_row.clear();
		//j = 0;
	}

	mat.populate(new_op);
}

// DONE
State op_rhs(const Operator& A, const State& psi)
{
	vector<complex<double>> resultant_vector;
	complex<double> res_psi_element = 0.0;

	vector<vector<complex<double>>>::const_iterator mat_row_it = A.get_start_address();

	vector<complex<double>>::const_iterator mat_col_it;
	vector<complex<double>>::const_iterator psi_col_it;

	for (; mat_row_it != A.get_end_address(); mat_row_it++)
	{
		mat_col_it = mat_row_it->begin();
		psi_col_it = psi.get_start_address();

		res_psi_element = 0.0;

		for (; mat_col_it != mat_row_it->end(), psi_col_it != psi.get_end_address(); mat_col_it++, psi_col_it++)
		{
			res_psi_element += ((*mat_col_it) * (*psi_col_it));
		}
		resultant_vector.push_back(res_psi_element);
	}

	State resultant_state = State(resultant_vector);

	return resultant_state;
}

// DONE?
/*
complex<double> inner_product_w_operator(const State& psi_1, const State& psi_2, const Operator& A)
{
	complex<double> l_inner_product = inner_product(psi_1, op_rhs(A, psi_2));

	return l_inner_product;
}
*/

// Two vectors are orthogonal if the inner product between the two is 0
/*
bool is_orthogonal(const State& psi_1, const State& psi_2)
{
	complex<double> l_inner_product = inner_product(psi_1, psi_2);

	return ((std::real(l_inner_product) == 0) && (std::imag(l_inner_product) == 0));
}
*/

/*
State tensor_product(State& psi_1, State& psi_2)
{

}
*/


void QR_decomposition(Operator& A, Operator& Q, Operator& R)
{
	Operator G(A.get_dim_x(), A.get_dim_y());
	Operator GT(A.get_dim_x(), A.get_dim_y());
	R = A;

	//vector<Operator> G_k;
	
	Q.createIdentityMatrix();

	unsigned int A_DIM_X = A.get_dim_x();
	unsigned int A_DIM_Y = A.get_dim_y();

	unsigned int A_DIM_X_ADJ = A_DIM_X - 1;
	unsigned int row = 0;
	// Determine which elements need to be zeroed out
	
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	for (unsigned int k = 0; k < A_DIM_X_ADJ; k++)
	{
		row = k + 1;
		if (!iszero(A.get_element(row, k)))
		{
			G.createIdentityMatrix();
			Givens_rotation_matrix(row, k, R.get_element(k, k), R.get_element(row, k), G);

			GT = G;
			GT.hermitian_congugate();

			//G.print();
			//GT.print();
			//system("PAUSE");

			R = G * R;

			Q = Q * GT;
		}
	}
	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;


	/*
	for (unsigned int row = 0; row < A_DIM_X; row++)
	{
		//cout << "ROW: " << row << endl;
		for (unsigned int col = 0; col < row; col++)
		{
			if (!iszero(A.get_element(row, col)))
			{
				//cout << "(ROW, COL) = (" << row << ", " << col << ")" << endl;
				
				
				//cout << "GIVENS ROTATION: ROW = " << row << ", COL = " << col << endl;
				
				G.createIdentityMatrix();
				Givens_rotation_matrix(row, col, R.get_element(col, col), R.get_element(row, col), G);

				GT = G;
				GT.hermitian_congugate();
				
				//G.print();
				//GT.print();
				//system("PAUSE");

				R = G * R;

				Q = Q * GT;

				//Q *= GT;
				

				//cout << "[ROW, COL] = [" << row << ", " << col << "], TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

				//G_k.push_back(GT);

				//Givens_rotation(A.get_element(col, col), A.get_element(row, col), G_k);
				//Givens_Rotation_Matrix.push_back(G_k);
			}
			//Givens_rotation(A.get_dim_x(), A.get_dim_y(), row, col, 90);
			//cout << this->m_matrix.at(row).at(col);
		}
		//cout << endl;
	}
	*/
	

	//Q.print();
	
	/*int iteration = 0;
	for (vector<Operator>::reverse_iterator it = G_k.rbegin(); it != G_k.rend(); it++)
	{
		//cout << "ITERATION: " << iteration << endl;
		//cout << "G" << endl;
		//it->print();
		//it->transpose();
		//it->print();
		Q = (*it) * Q;
		//cout << "Q" << endl;
		//Q.print();
		iteration++;
	}*/
	
	return;
}

/*

void Givens_rotation(unsigned int dim_x, unsigned int dim_y, complex<double> a, complex<double> b)
{
	// Norm = Norm Squared...

	// mag = sqrt(|a|^2 + |b|^2)
	double magnitude = std::sqrt(norm(a) + norm(b));
	double inverse_magnitude = 1.0 / magnitude;

	complex<double> cosine = a * inverse_magnitude;
	complex<double> sine = b * inverse_magnitude;

	// Create local 2D matrix of dimension dim_x * dim_y (SHOULD JUST BE SQUARE?)
	vector<vector<complex<double>>> G(dim_x, vector<complex<double>>(dim_y, 0.0));

	// Set diagonal elements
	for (int k = 0; k < dim_x; k++)
	{
		if (k == row || k == col)
		{
			G.at(k).at(k) = std::cos(theta);
		}
		else if (k != row && k != col)
		{
			G.at(k).at(k) = 1;
		}
	}

	G.at(row).at(col) = std::sin(theta);
	G.at(col).at(row) = -1 * std::sin(theta);

	cout << "GIVENS ROTATION (" << row << ", " << col << ")" << endl;

	for (int i = 0; i < dim_x; i++)
	{
		for (int j = 0; j < dim_y; j++)
		{
			cout << G.at(i).at(j);
		}
		cout << endl;
	}

} */

double sub_diagonal_error(const Operator& A, const Operator& B)
{
	long double error_sq = 0.0;
	complex<double> A_ELEM = 0.0;
	complex<double> B_ELEM = 0.0;

	int A_DIM = A.get_dim_x();

	for (int k = 2; k < A_DIM; k++)
	{
		A_ELEM = A.get_element(k, k - 2);
		B_ELEM = B.get_element(k, k - 2);

		if (iszero(A_ELEM))
		{
			A_ELEM = 0.0;
		}

		if (iszero(B_ELEM))
		{
			B_ELEM = 0.0;
		}

		error_sq += std::abs(A_ELEM - B_ELEM);
	}
	return error_sq;
}

double diagonal_error(const Operator& A, const Operator& B)
{
	double error_sq = 0.0;

	complex<double> det_A_subblock = 0.0;
	complex<double> det_B_subblock = 0.0;

	int A_DIM = A.get_dim_x();
	// Iterate over the first dim_x - 1 elements - avoid out of bound errors at the last element
	for (int k = 0; k < A_DIM  - 1; k++)
	{
		
		// If off-diagonal element exists, then current diagonal element is part of larger block - complex eigenvalue

		// 1D block in both A and B
		if (std::norm(A.get_element(k + 1, k)) < QR_ALG_ERROR && std::norm(B.get_element(k + 1, k)) < QR_ALG_ERROR)
		{
			//cout << "1D" << endl;
			error_sq += (std::norm(A.get_element(k, k) - B.get_element(k, k)));
		}

		// Else a 2D block was encountered in A or B
		else
		{
			det_A_subblock = ((A.get_element(k, k) * A.get_element(k + 1, k + 1)) - (A.get_element(k, k + 1) * A.get_element(k + 1, k)));
			det_B_subblock = ((B.get_element(k, k) * B.get_element(k + 1, k + 1)) - (B.get_element(k, k + 1) * B.get_element(k + 1, k)));
			//cout << "2D" << endl;
			//cout << "2D" << endl;
			// Extract the 2D block
			//Operator A_SUBBLOCK ({ {A.get_element(k, k), A.get_element(k, k + 1)}, {A.get_element(k + 1, k), A.get_element(k + 1, k + 1)} });
			//Operator B_SUBBLOCK({ {B.get_element(k, k), B.get_element(k, k + 1)}, {B.get_element(k + 1, k), B.get_element(k + 1, k + 1)} });

			error_sq += std::norm(det_A_subblock - det_B_subblock);
			k++;
		}
	}
	//cout << error << endl;
	return error_sq;
}

double approx_diagonal(const Operator& A, const Operator& B)
{
	double error = 0.0;
	complex<double> det_A_sub_mat = 0.0;
	complex<double> det_B_sub_mat = 0.0;

	for (unsigned int k = 0; k < A.get_dim_x() - 1; k++)
	{
		det_A_sub_mat = (A.get_element(k, k) * A.get_element(k + 1, k + 1)) - (A.get_element(k, k + 1) * A.get_element(k + 1, k));
		det_B_sub_mat = (B.get_element(k, k) * B.get_element(k + 1, k + 1)) - (B.get_element(k, k + 1) * B.get_element(k + 1, k));


		error += std::abs(det_A_sub_mat - det_B_sub_mat);
	}
	return error;
}

vector<complex<double>> eigval_mat2(Operator& mat)
{
	// Get the elements for the characteristic polynomial
	complex<double> b = -1.0 * (mat.get_element(0, 0) + mat.get_element(1, 1));
	complex<double> c = (mat.get_element(0, 0) * mat.get_element(1, 1)) - (mat.get_element(0, 1) * mat.get_element(1, 0));

	// Solve for the roots of the characteristic polynomial using the quadratic formula
	complex<double> root = std::sqrt(b * b - (4.0 * c));
	complex<double> lambda_plus = (((-1.0 * b) + root) / 2.0);
	complex<double> lambda_minus = (((-1.0 * b) - root) / 2.0);

	// Return vector of eigenvalues
	return { lambda_plus, lambda_minus };
}

vector<complex<double>> eigval_mat3(Operator& mat)
{
	return { 0, 0, 0 };
}

// First convert A to Hessenberg form - H_A = Q^T * A * Q
void QR_Algorithm(Operator& A, Operator& A_decomposition, vector<complex<double>>& A_eigenvalues)
{
	Operator A_PREV = A;
	Operator A_CURR = A;
	Operator Q(A.get_dim_x(), A.get_dim_y());
	Operator R(A.get_dim_x(), A.get_dim_y());

	double error_sq = 10;

	unsigned int iteration = 0;

	//while (error_sq >= 1e-20 && iteration < QR_ALG_MAX_IT)
	while (error_sq >= 1e-10 && iteration < QR_ALG_MAX_IT)
	{
		//cout << "ITERATION: " << iteration << endl;
		//std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
		
		//std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
		QR_decomposition(A_PREV, Q, R);
		//std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
		//cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
		
		
		A_CURR = R * Q;
		
		
		//std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

		//A_CURR.print();
		//system("PAUSE");


		
		error_sq = diagonal_error(A_CURR, A_PREV);
		
		if (iteration % 10 == 0)
		{
			cout << "ITERATION = " << iteration << ", ERROR = " << error_sq << endl;
			//A_CURR.print_shape();
		}
		

		//cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
		//A_CURR.print();
		
		//A_PREV.populate(A_CURR.get_matrix());// = A_CURR;
		A_PREV = A_CURR;

		iteration++;

		
		
		//if (iteration % 10 == 0)
		//{
			
		//}
	}

	// Get the eigenvalues - in blocks along the diagonal
	
	// Iterate over the first dim_x - 1 elements - avoid out of bound errors at the last element
	for (int k = 0; k < A_CURR.get_dim_x() - 1; k++)
	{
		// If off-diagonal element exists, then current diagonal element is part of larger block - complex eigenvalue

		// If the off-diagonal element is sufficiently large, then a 2D block was encountered
		if (std::abs(A_CURR.get_element(k + 1, k)) > (10e15 * QR_ALG_ERROR))
		{
			// Extract the 2D block
			Operator BLOCK_2D({ {A_CURR.get_element(k, k), A_CURR.get_element(k, k + 1)}, {A_CURR.get_element(k + 1, k), A_CURR.get_element(k + 1, k + 1)} });

			vector<complex<double>> block_eigvals = eigval_mat2(BLOCK_2D);
			
			if (!A_eigenvalues.empty())
			{
				A_eigenvalues.push_back(block_eigvals.at(0));
				A_eigenvalues.push_back(block_eigvals.at(1));
			}
			else
			{
				A_eigenvalues = block_eigvals;
			}
			k++;
		}
		// Else the current diagonal element is part of a block of 1 so it is real eigenvalue - append to list
		else
		{
			A_eigenvalues.push_back(A_CURR.get_element(k, k));
		}
	}

	// May need to add the last eigenvalue - would only need to do this if matrix contained only 1x1 blocks
	if (A_eigenvalues.size() < A.get_dim_x())
	{
		A_eigenvalues.push_back(A_CURR.get_element(A.get_dim_x() - 1, A.get_dim_y() - 1));
	}


	cout << "EIGENVALUE DECOMPOSITION COMPLETE" << endl;
	cout << "NUMBER OF ITERATIONS: " << iteration << endl;
	cout << "ERROR: " << error_sq << endl;

	for (int row = 0; row < A_CURR.get_dim_x(); row++)
	{
		for (int col = 0; col < A_CURR.get_dim_y(); col++)
		{
			//cout << std::abs(A.get_element(row, col)) << endl;
			if (std::abs(A_CURR.get_element(row, col)) <= 0.00005)
			{
				cout << "0 ";
			}
			else if (row <= col)
			{
				cout << "* ";
			}
			else if (row > col)
			{
				cout << "+ ";
			}
		}
		cout << endl;
	}


	//A_CURR.print();
	A_decomposition = A_CURR;
}

void Givens_rotation_matrix(const unsigned int row, const unsigned int col, const complex<double>& a, const complex<double>& b, Operator& G)
{
	complex<double> c = 0.0;
	complex<double> s = 0.0;
	double r = 0.0;

	if (iszero(b))
	{
		c = 1.0;
		s = 0;
		r = std::abs(a);
	}
	else if (iszero(a))
	{
		c = 0.0;
		s = sign(std::conj(b));
		r = std::abs(b);
	}
	else if (!iszero(a) && !iszero(b))
	{
		r = std::sqrt(std::norm(a) + std::norm(b));

		c = std::abs(a) / r;
		s = (sign(a) * std::conj(b)) / r;
	}

	G.set_element(col, col, c);
	G.set_element(row, row, c);

	G.set_element(col, row, 1.0 * s);
	G.set_element(row, col, -1.0 * std::conj(s));

	//vector<vector<complex<double>>> G_elements(dim_x, vector<complex<double>>(dim_y, 0.0));

	//r = std::sqrt(std::norm(a) + std::norm(b));

	//c = a / r;
	//s = -1.0 * (b / r);

	

	// NEW
	

	// OLD

	//G.set_element(col, row, -1.0 * std::conj(s));
	//G.set_element(row, col, s);

	/*// Set diagonal elements
	for (unsigned int k = 0; k < dim_x; k++)
	{
		if (k == col)
		{
			G_elements.at(k).at(k) = std::conj(c);
		}
		else if (k == row)
		{
			G_elements.at(k).at(k) = c;
		}
		else if (k != row && k != col)
		{
			G_elements.at(k).at(k) = 1.0;
		}
	}

	G_elements.at(col).at(row) = -1.0 * std::conj(s);
	G_elements.at(row).at(col) = s;

	G.populate(G_elements);
	*/
	return;
}

void Givens_rotation(const complex<double>& a, const complex<double>& b, Givens_rotation_t& Gij)
{
	complex<double> c = 0.0;
	complex<double> s = 0.0;

	double r = 0.0;

	r = std::sqrt(std::norm(a) + std::norm(b));

	c = a / r;
	s = -1.0 * (b / r);

	Gij.c = c;
	Gij.s = s;
	Gij.radius = r;

	return;
}


// Algorithm adapted from Wikipedia and source <>
// TODO: Add special cases for optimizations
vector<vector<complex<double>>> Givens_rotation(unsigned int dim_x, unsigned int dim_y, unsigned int row, unsigned int col, complex<double> a, complex<double> b)
{
	complex<double> c = 0.0;
	complex<double> s = 0.0;
	double r = 0.0;

	vector<vector<complex<double>>> G(dim_x, vector<complex<double>>(dim_y, 0.0));

	r = std::sqrt(std::norm(a) + std::norm(b));

	cout << "a = " << a << endl;
	cout << "b = " << b << endl;

	c = a / r;

	s = -1.0 * (b / r);

	cout << "ROW = " << row << endl;
	cout << "COL = " << col << endl;
	cout << "A[" << row << ", " << col << "] = " << b << endl;

	cout << "c = " << c << endl;
	cout << "s = " << s << endl;
	cout << "r = " << r << endl;


	// Set diagonal elements
	for (unsigned int k = 0; k < dim_x; k++)
	{
		if (k == col)
		{
			G.at(k).at(k) = std::conj(c);
		}
		else if (k == row)
		{
			G.at(k).at(k) = c;
		}
		else if (k != row && k != col)
		{
			G.at(k).at(k) = 1.0;
		}		
	}

	G.at(col).at(row) = -1.0 * std::conj(s);
	G.at(row).at(col) = s;

	return G;
}

// Via Householder Reflection
void Hessenberg_reduction(Operator& H)
{
	//H = A;

	//vector<complex<double>> x;
	//vector<complex<double>> v;// (x.size(), 0.0);

	double x_mag_sq = 0.0;
	double coeff = 0.0;
	complex<double> v0 = 0.0;

	unsigned int DIM = H.get_dim_x();
	unsigned int DIM_IT = DIM - 2;

	Operator v_mat;
	Operator SUBMAT_HORIZONTAL;
	Operator SUBMAT_VERTICAL;

	std::chrono::steady_clock::time_point global_start_time = std::chrono::steady_clock::now();
	// Overall loop 
	for (unsigned int k = 0; k < DIM_IT; k++)
	{
		cout << "HESSENBERG REDUCTION ITERATION: " << k << endl;
		
		std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
		
		vector<complex<double>> x;
		//x.clear();
		x_mag_sq = 0.0;
		//cout << "INDEX = " << k << endl;
		// Extract the column

		for (unsigned int i = k + 1; i < DIM; i++)
		{
			x.push_back(H.get_element(i, k));
			x_mag_sq += std::norm(x[i - k - 1]);
		}

		std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

		cout << "EXTRACT COLUMN TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

		// Find magnitude of x
		//for (int i = 0; i < x.size(); i++)
		//{
			//x_mag_sq += std::norm(x[i]);
		//}

		// Compute the unit vector v
		//vector<complex<double>> v(x.size(), 0.0);
		//v0 = ;
		start_time = std::chrono::steady_clock::now();

		v0 = (x[0] / std::abs(x[0])) * std::sqrt(x_mag_sq);
		x_mag_sq += std::real((x[0] * std::conj(v0) + std::conj(x[0]) * v0 + std::norm(v0)));
		x[0] += v0;
		//std::transform(src1_begin, src1_end, src2_start, dest, op)
		//std::transform(v.begin(), v.end(), x.begin(), v.begin(), std::plus<complex<double>>());

		//x_mag_sq += ;

		//x_mag_sq = 0.0;
		//for (int j = 0; j < x.size(); j++)
		//{
			//x_mag_sq += std::norm(x[j]);
		//}
		coeff = (2.0 / x_mag_sq);

		stop_time = std::chrono::steady_clock::now();
		cout << "UNIT VECTOR TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
		//inv_v_mag = (1.0 / std::sqrt(v_mag_sq));

		// Normalize unit vector v
		//std::transform(v.begin(), v.end(), v.begin(), std::bind1st(std::multiplies<complex<double>>(), inv_v_mag));

		// Get the submatrices
	
		
		start_time = std::chrono::steady_clock::now();
		// Calculate the matrix adjustment from the unit vector v
		v_mat.set_dim(x.size(), x.size());

		outer_product(v_mat, x, x);

		stop_time = std::chrono::steady_clock::now();
		cout << "OUTER PRODUCT TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

		//cout << "V MAT" << endl;
		//v_mat.print();
		//system("PAUSE");

		//cout << "HORIZONTAL" << endl;
		//SUBMAT_HORIZONTAL.print();
		//cout << "VERTICAL" << endl;
		//SUBMAT_VERTICAL.print();
		//cout << "OUTER PRODUCT" << endl;
		//v_mat.print();
		//system("PAUSE");

		// Check ordering

		start_time = std::chrono::steady_clock::now();
		SUBMAT_HORIZONTAL = H.get_submatrix(k + 1, DIM - 1, k, DIM - 1);
		SUBMAT_HORIZONTAL = SUBMAT_HORIZONTAL - ((v_mat * SUBMAT_HORIZONTAL) * coeff);
		H.set_submatrix(k + 1, DIM - 1, k, DIM - 1, SUBMAT_HORIZONTAL.get_matrix());
		stop_time = std::chrono::steady_clock::now();
		cout << "HORIZONTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
		//H.print();
		//system("PAUSE");

		start_time = std::chrono::steady_clock::now();
		SUBMAT_VERTICAL = H.get_submatrix(0, DIM - 1, k + 1, DIM - 1);
		SUBMAT_VERTICAL = SUBMAT_VERTICAL - ((SUBMAT_VERTICAL * v_mat) * coeff);
		H.set_submatrix(0, DIM - 1, k + 1, DIM - 1, SUBMAT_VERTICAL.get_matrix());
		stop_time = std::chrono::steady_clock::now();
		cout << "VERTICAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

		

		

		//cout << "HORIZONTAL" << endl;
		//SUBMAT_HORIZONTAL.print();
		//cout << "VERTICAL" << endl;
		//SUBMAT_VERTICAL.print();

	}
	std::chrono::steady_clock::time_point global_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(global_stop_time - global_start_time).count() << " ms" << std::endl;

	//H.print();


	/*
	for (unsigned int k = )

	unsigned int A_DIM_X = A.get_dim_x();
	unsigned int A_DIM_Y = A.get_dim_y();
	Operator G(A.get_dim_x(), A.get_dim_y());

	H = A;

	vector<Operator> G_k;

	for (unsigned int row = 2; row < A_DIM_X; row++)
	{
		for (unsigned int col = 0; col < row - 1; col++)
		{
			// If element is not zero apply Givens rotations
			if (A.get_element(row, col) != 0.0)
			{
				G.createIdentityMatrix();
				Givens_rotation_matrix(A_DIM_X, A_DIM_Y, row, col, H.get_element(col, col), H.get_element(row, col), G);

				H = G * H;
			}
		}
	}
	*/
	return;
}

void Strassen_matrix_multiplication(Operator& C, Operator& A, Operator& B)
{
	// Assume power of 2...

	// A11 ------------------------
	unsigned int row_a11_start = 0;
	unsigned int row_a11_stop = A.get_dim_x() / 2 - 1;

	unsigned int col_a11_start = 0;
	unsigned int col_a11_stop = A.get_dim_y() / 2 - 1;

	// A12 ------------------------
	unsigned int row_a12_start = 0;
	unsigned int row_a12_stop = A.get_dim_x() / 2 - 1;

	unsigned int col_a12_start = A.get_dim_y() / 2;
	unsigned int col_a12_stop = A.get_dim_y() - 1;

	// A21 ------------------------
	unsigned int row_a21_start = A.get_dim_x() / 2;
	unsigned int row_a21_stop = A.get_dim_x() - 1;

	unsigned int col_a21_start = 0;
	unsigned int col_a21_stop = A.get_dim_y() / 2 - 1;

	// A22 ------------------------
	unsigned int row_a22_start = A.get_dim_x() / 2;
	unsigned int row_a22_stop = A.get_dim_x() - 1;

	unsigned int col_a22_start = A.get_dim_y() / 2;
	unsigned int col_a22_stop = A.get_dim_y() - 1;

	// B11 ------------------------
	unsigned int row_b11_start = 0;
	unsigned int row_b11_stop = B.get_dim_x() / 2 - 1;

	unsigned int col_b11_start = 0;
	unsigned int col_b11_stop = B.get_dim_y() / 2 - 1;

	// B12 ------------------------
	unsigned int row_b12_start = 0;
	unsigned int row_b12_stop = B.get_dim_x() / 2 - 1;

	unsigned int col_b12_start = B.get_dim_y() / 2;
	unsigned int col_b12_stop = B.get_dim_y() - 1;

	// B21 ------------------------
	unsigned int row_b21_start = B.get_dim_x() / 2;
	unsigned int row_b21_stop = B.get_dim_x() - 1;

	unsigned int col_b21_start = 0;
	unsigned int col_b21_stop = B.get_dim_y() / 2 - 1;

	// B22 ------------------------
	unsigned int row_b22_start = B.get_dim_x() / 2;
	unsigned int row_b22_stop = B.get_dim_x() - 1;

	unsigned int col_b22_start = B.get_dim_y() / 2;
	unsigned int col_b22_stop = B.get_dim_y() - 1;

	Operator A11 = A.get_submatrix(row_a11_start, row_a11_stop, col_a11_start, col_a11_stop);
	Operator A12 = A.get_submatrix(row_a12_start, row_a12_stop, col_a12_start, col_a12_stop);
	Operator A21 = A.get_submatrix(row_a21_start, row_a21_stop, col_a21_start, col_a21_stop);
	Operator A22 = A.get_submatrix(row_a22_start, row_a22_stop, col_a22_start, col_a22_stop);

	cout << "A SUBMATRICES COMPLETE" << endl;

	Operator B11 = B.get_submatrix(row_b11_start, row_b11_stop, col_b11_start, col_b11_stop);
	Operator B12 = B.get_submatrix(row_b12_start, row_b12_stop, col_b12_start, col_b12_stop);
	Operator B21 = B.get_submatrix(row_b21_start, row_b21_stop, col_b21_start, col_b21_stop);
	Operator B22 = B.get_submatrix(row_b22_start, row_b22_stop, col_b22_start, col_b22_stop);
	
	cout << "B SUBMATRICES COMPLETE" << endl;

	Operator M1 = (A11 + A11) * (B11 + B22);
	cout << "M1 SUBMATRIX COMPLETE" << endl;

	Operator M2 = (A21 + A22) * B11;
	cout << "M2 SUBMATRIX COMPLETE" << endl;

	Operator M3 = A11 * (B12 - B22);
	cout << "M3 SUBMATRIX COMPLETE" << endl;

	Operator M4 = A22 * (B21 - B11);
	cout << "M4 SUBMATRIX COMPLETE" << endl;

	Operator M5 = (A11 + A12) * B22;
	cout << "M5 SUBMATRIX COMPLETE" << endl;

	Operator M6 = (A21 - A11) * (B11 + B12);
	cout << "M6 SUBMATRIX COMPLETE" << endl;

	Operator M7 = (A12 - A22) * (B21 + B22);
	cout << "M7 SUBMATRIX COMPLETE" << endl;

	Operator C11 = M1 + M4 - M5 + M7;
	Operator C12 = M3 + M5;
	Operator C21 = M2 + M4;
	Operator C22 = M1 - M2 + M3 + M6;

	cout << "C SUBMATRICES COMPLETE" << endl;

	//A11.print();
	//A12.print();
	//A21.print();
	//A22.print();

	//B11.print();
	//B12.print();
	//B21.print();
	//B22.print();

	//C11.print();
	//C12.print();
	//C21.print();
	//C22.print();





}
#endif

