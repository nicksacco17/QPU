
#include <iostream>
#include <iomanip>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <functional>
#include "Operator.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;
using std::string;

Operator::Operator()
{
	m_dim_x = 2;
	m_dim_y = 2;
	m_trace = 0.0;
	m_determinant = 0.0;

	m_matrix = { {0, 0}, {0, 0} };
}

Operator::Operator(unsigned int in_size_x, unsigned int in_size_y)
{
	m_dim_x = in_size_x;
	m_dim_y = in_size_y;
	m_trace = 0.0;
	m_determinant = 0.0;

	m_matrix.clear();
	
	m_matrix = vector<vector<complex<double>>>(m_dim_x, vector<complex<double>>(m_dim_y, 0.0));

	//for (unsigned int i = 0; i < m_dim_x; i++)
	//{
		//std::fill(m_matrix[i].begin(), m_matrix[i].end(), 0.0);
		//m_matrix.push_back(vector<complex<double>>(m_dim_y, 0.0));
	//}
}

Operator::Operator(const vector<vector<complex<double>>>& in_matrix)
{
	m_matrix = in_matrix;
	m_dim_x = in_matrix.size();
	m_dim_y = in_matrix.at(0).size();

	//i_calc_trace();
	//i_calc_determinant();
}

Operator::~Operator()
{
	m_matrix.clear();
}

unsigned int Operator::get_dim_x() const
{
	return m_dim_x;
}

unsigned int Operator::get_dim_y() const
{
	return m_dim_y;
}

complex<double> Operator::get_trace() const
{
	return m_trace;
}

complex<double> Operator::get_determinant() const
{
	return m_determinant;
}

/*complex<double> Operator::get_element(int row, int col) const
{
	return (m_matrix.at(row)).at(col);
}*/

void Operator::set_element(int row, int col, complex<double> val)
{
	m_matrix.at(row).at(col) = val;
}

vector<vector<complex<double>>> Operator::get_matrix() const
{
	return m_matrix;
}

void Operator::populate(const vector<vector<complex<double>>>& in_matrix)
{
	m_matrix = in_matrix;
	m_dim_x = in_matrix.size();
	m_dim_y = in_matrix.at(0).size();

	i_calc_trace();
	i_calc_determinant();
}

void Operator::createIdentityMatrix()
{
	for (unsigned int i = 0; i < m_dim_x; i++)
	{
		std::fill(m_matrix[i].begin(), m_matrix[i].end(), 0.0);
		m_matrix[i][i] = 1.0;
		//vector<complex<double>> row(m_dim_y, 0.0);
		//row.at(i) = 1.0;
		//m_matrix.push_back(row);
	}

	m_trace = m_dim_x;
	m_determinant = 1;
}

void Operator::print() const
{
	cout << "---------- PRINT OPERATOR ----------" << endl;

	cout << "DIMENSION: (" << m_dim_x << " x " << m_dim_y << ")" << endl;
	cout << "TRACE: " << m_trace << endl;
	cout << "DETERMINANT: " << m_determinant << endl;

	//cout << "SQUARE: " << std::boolalpha << is_square() << endl;
	//cout << "HERMITIAN: " << std::boolalpha << is_Hermitian() << endl;
	//cout << "UNITARY: " << std::boolalpha << is_Unitary() << endl;
	cout << "ELEMENTS:\n" << endl;

	vector<vector<complex<double>>>::const_iterator row_it;
	vector<complex<double>>::const_iterator col_it;

	for (row_it = m_matrix.begin(); row_it != m_matrix.end(); row_it++)
	{
		cout << "| ";
		for (col_it = row_it->begin(); col_it != row_it->end(); col_it++)
		{
			cout << std::fixed << std::setprecision(6) << std::setfill(' ') << *col_it;
		}
		cout << "|" << endl;
	}

	cout << "\n---------- PRINT OPERATOR ----------" << endl;
}

void Operator::print_shape() const
{
	for (int i = 0; i < m_dim_x; i++)
	{
		for (int j = 0; j < m_dim_y; j++)
		{
			if (std::abs(m_matrix.at(i).at(j)) <= 1e-10)
			{
				cout << "0 ";
			}
			else if (i == j)
			{
				cout << "\\ ";
			}
			else
			{
				cout << "* ";
			}
		}
		cout << endl;
	}


}

// Return true if the matrix is a square (n x n) matrix
bool Operator::is_Square()
{
	return (m_dim_x == m_dim_y);
}

// Singular matrix --> No inverse
// If not square --> Treat as singular
// If square but determinant == 0 --> singular
// Return true if the matrix is singular (i.e. det = 0)
bool Operator::is_Singular()
{
	bool singular_flag = true;

	// If square matrix and determinant is not 0, then it is non-singular
	if (m_dim_x == m_dim_y && !iszero(m_determinant))
	{
		return false;
	}

	return singular_flag;
}

// Return true if the matrix is a Hermitian matrix (A = A*T)
// Requires square matrix
bool Operator::is_Hermitian()
{
	bool is_hermitian_flag = true;

	// Only check if square matrix
	if (m_dim_x == m_dim_y)
	{
		// First check the diagonal - in Hermitian matrices, the diagonal should be REAL
		for (unsigned int k = 0; k < m_dim_x; k++)
		{
			// If any elements along the diagonal are not real, return false and stop checking
			if (!iszero(std::imag(m_matrix[k][k])))
			{
				is_hermitian_flag = false;
			}
		}

		// If all the elements along the diagonal are REAL, check that the rest of the matrix
		// satisfies the Hermitian requirements: Aij = Aji*
		for (unsigned int row = 0; row < m_dim_x - 1, is_hermitian_flag == true; row++)
		{
			for (unsigned int col = row + 1; col < m_dim_y, is_hermitian_flag == true; col++)
			{
				if (m_matrix[row][col] != std::conj(m_matrix[col][row]))
				{
					is_hermitian_flag = false;
				}
			}
		}

		return true;
	}

	return false;
}

// Return true if the matrix is a Unitary matrix (AA*T = A*TA = I)
// Requires square matrix
bool Operator::is_Unitary()
{
	Operator RESULT;
	
	Operator IDENTITY(m_dim_x, m_dim_y);
	IDENTITY.createIdentityMatrix();

	Operator A_DAGGER = *this;
	A_DAGGER.hermitian_congugate();

	// First try A * A_DAG
	RESULT = *this * A_DAGGER;

	// If the result is Identity matrix, continue testing 
	if (RESULT == IDENTITY)
	{
		// Then try A_DAG * A
		RESULT = A_DAGGER * *this;

		// If the result is Identity matrix, original matrix A is unitary

		if (RESULT == IDENTITY)
		{
			return true;
		}
	}

	
	

	return true;
}





// Can only calculate trace on square matrices
void Operator::i_calc_trace()
{
	m_trace = 0;
	if (m_dim_x == m_dim_y)
	{
		for (unsigned int i = 0; i < m_dim_x; i++)
		{
			m_trace += m_matrix.at(i).at(i);
		}
	}
	
}

/* ************************************* OPERATORS ************************************** */

// Assignment (A = B)
Operator& Operator::operator=(const Operator& mat)
{
	if (this != &mat)
	{
		this->m_dim_x = mat.m_dim_x;
		this->m_dim_y = mat.m_dim_y;
		this->m_matrix = mat.m_matrix;

		//this->i_calc_determinant();
		//this->i_calc_trace();
	}

	return *this;
}

// Equality (A == B)
bool Operator::operator==(const Operator& mat)
{
	bool is_equal = true;

	// If the two matrices have the same dimension, then they can be compared
	if (this->m_dim_x == mat.m_dim_x && this->m_dim_y == mat.m_dim_y)
	{
		vector<vector<complex<double>>>::iterator lhs_it = this->m_matrix.begin();
		vector<vector<complex<double>>>::const_iterator rhs_it = mat.m_matrix.begin();

		// Not sure if this loop construction is allowed - have had issues using the get() functions
		for (; is_equal, lhs_it != this->m_matrix.end(), rhs_it != mat.m_matrix.end(); lhs_it++, rhs_it++)
		{
			if (*lhs_it != *rhs_it)
			{
				is_equal = false;
			}
		}
	}
	else
	{
		is_equal = false;
	}

	return is_equal;
}

// Not Equal (A != B)
bool Operator::operator!=(const Operator& mat)
{
	bool is_equal = (*this == mat);
	return !is_equal;
}

// Addition (A += B)
Operator& Operator::operator+=(const Operator& mat)
{
	if (this != &mat && this->m_dim_x == mat.m_dim_x && this->m_dim_y == mat.m_dim_y)
	{

		//std::transform(src1_begin, src1_end, src2_start, dest, op)
		//std::transform(this->m_matrix.begin(), this->m_matrix.end(), mat.m_matrix.begin(), this->m_matrix.begin(), std::plus<complex<double>>());

		vector<vector<complex<double>>>::iterator lhs_row_it = this->m_matrix.begin();
		vector<vector<complex<double>>>::const_iterator rhs_row_it = mat.m_matrix.begin();

		vector<complex<double>>::iterator lhs_col_it;
		vector<complex<double>>::const_iterator rhs_col_it;

		for (; lhs_row_it != this->m_matrix.end(), rhs_row_it != mat.m_matrix.end(); lhs_row_it++, rhs_row_it++)
		{
			lhs_col_it = lhs_row_it->begin();
			rhs_col_it = rhs_row_it->begin();

			for (; lhs_col_it != lhs_row_it->end(), rhs_col_it != rhs_row_it->end(); lhs_col_it++, rhs_col_it++)
			{
				*lhs_col_it += *rhs_col_it;
			}
		}
	}

	return *this;
}

// Addition (C = A + B)
const Operator Operator::operator+(const Operator& mat) const
{
	Operator mat_intermediate = *this;
	mat_intermediate += mat;

	return mat_intermediate;
}

// Subtraction (A -= B)
Operator& Operator::operator-=(const Operator& mat)
{
	if (this != &mat && this->m_dim_x == mat.m_dim_x && this->m_dim_y == mat.m_dim_y)
	{
		//std::transform(src1_begin, src1_end, src2_start, dest, op)
		//std::transform(this->m_matrix.begin(), this->m_matrix.end(), mat.m_matrix.begin(), this->m_matrix.begin(), std::minus<complex<double>>());
		
		for (int i = 0; i < m_dim_x; i++)
		{
			for (int j = 0; j < m_dim_y; j++)
			{
				this->m_matrix[i][j] -= mat.m_matrix[i][j];
			}
		}
		
		/*vector<vector<complex<double>>>::iterator lhs_row_it = this->m_matrix.begin();
		vector<vector<complex<double>>>::const_iterator rhs_row_it = mat.m_matrix.begin();

		vector<complex<double>>::iterator lhs_col_it;
		vector<complex<double>>::const_iterator rhs_col_it;

		for (; lhs_row_it != this->m_matrix.end(), rhs_row_it != mat.m_matrix.end(); lhs_row_it++, rhs_row_it++)
		{
			lhs_col_it = lhs_row_it->begin();
			rhs_col_it = rhs_row_it->begin();

			for (; lhs_col_it != lhs_row_it->end(), rhs_col_it != rhs_row_it->end(); lhs_col_it++, rhs_col_it++)
			{
				*lhs_col_it -= *rhs_col_it;
			}
		}*/
	}

	return *this;
}

// Subtraction (C = A - B)
const Operator Operator::operator-(const Operator& mat) const
{
	Operator mat_intermediate = *this;
	mat_intermediate -= mat;

	return mat_intermediate;
}

// Multiplication (A *= a)
Operator& Operator::operator*=(const complex<double> a)
{
	vector<vector<complex<double>>>::iterator mat_row_it = this->m_matrix.begin();
	vector <complex<double>>::iterator mat_col_it;

	for (; mat_row_it != this->m_matrix.end(); mat_row_it++)
	{
		for (mat_col_it = mat_row_it->begin(); mat_col_it != mat_row_it->end(); mat_col_it++)
		{
			*mat_col_it *= a;
		}
	}

	return *this;
}

// Multiplication (B = A * a)
const Operator Operator::operator*(const complex<double> a) const
{
	Operator mat_intermediate = *this;
	mat_intermediate *= a;

	return mat_intermediate;
}

// Multiplication (A *= B)
Operator& Operator::operator*=(const Operator& mat)
{
	//Operator temp = mat;

	// (this_x, this_y) x (mat_x, mat_y)
	// Need this_y == mat_x --> Resultant matrix (this_x, mat_y)

	vector<vector<complex<double>>> new_mat(this->m_dim_x, vector<complex<double>>(mat.m_dim_y, 0.0));

	if (this != &mat && this->m_dim_y == mat.m_dim_x)
	{
		for (unsigned int i = 0; i < this->m_dim_x; i++)
		{
			for (unsigned int k = 0; k < this->m_dim_y; k++)
			{
				for (unsigned int j = 0; j < mat.m_dim_y; j++)
				{
					new_mat[i][j] += this->m_matrix[i][k] * mat.m_matrix[k][j];
				}
			}
		}

		/*


		complex<double> INIT = 0.0;
		complex<double> new_mat_elem = 0.0;

		//unsigned int new_dim_x = this->m_dim_x;
		//unsigned int new_dim_y = mat.m_dim_y;
		
		
		//new_mat.reserve(new_dim_x * new_dim_y);
		//vector<complex<double>> new_row;//(new_dim_y, 0.0);
		//new_row.reserve(new_dim_y);
		//vector<complex<double>> l_row(new_dim_x, 0.0);
		temp.transpose();

		unsigned int row = 0;
		unsigned int col = 0;

		for (vector<vector<complex<double>>>::const_iterator lhs_it = this->m_matrix.begin(); lhs_it != this->m_matrix.end(); lhs_it++)
		{
			//if (lhs_row % 10 == 0)
			//{
				//cout << "ROW: " << lhs_row << endl;
			//}
			//for (unsigned int rhs_row = 0; rhs_row < temp.m_dim_x; rhs_row++)
			col = 0;
			for (vector<vector<complex<double>>>::const_iterator rhs_it = temp.m_matrix.begin(); rhs_it != temp.m_matrix.end(); rhs_it++)
			{
				//std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
				//l_row = this->m_matrix[lhs_row];
				//new_mat_elem = std::inner_product(l_row.begin(), l_row.end(), temp.m_matrix[rhs_row].begin(), INIT);
				//new_mat_elem = std::inner_product(lhs_it->begin(), lhs_it->end(), rhs_it->begin(), INIT);
				new_mat[row][col] = std::inner_product(lhs_it->begin(), lhs_it->end(), rhs_it->begin(), INIT);
				//new_row.push_back(new_mat_elem);
				//new_mat_elem = std::inner_product(this->m_matrix[lhs_row].begin(), this->m_matrix[lhs_row].end(), temp.m_matrix[rhs_row].begin(), INIT);
				//cout << "NEW MAT ELEM " << new_mat_elem << endl;
				//this->m_matrix[lhs_row][rhs_row] = new_mat_elem;
				//new_mat[lhs_row][rhs_row] = new_mat_elem;
				//std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
				//std::cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() << " us" << std::endl;
				col++;
			}
			//new_mat.push_back(new_row);
			//new_row.clear();
			//lhs_row++;
			row++;
		}
		//for (unsigned int lhs_row = 0; lhs_row < this->m_dim_x; lhs_row++)
		//{
			
			
			//new_mat[lhs_row][k] = new_mat_elem;
			//new_mat_elem = 0.0;
			//k = (k + 1) % this->m_dim_x;
		//}

		
		for (unsigned int i = 0; i < new_dim_x; i++)
		{
			for (unsigned int k = 0; k < new_dim_x; k++)
				//for (unsigned int j = 0; j < N; j++)
			{
				//sum = 0.0;
				for (unsigned int j = 0; j < new_dim_x; j++)
					//for (unsigned int k = 0; k < N; k++)
				{
					//A_index = (i * N + k);
					//B_index = (j + k * N);

					new_mat[i][j] += (this->m_matrix[i][k] * mat.m_matrix[k][j]);
					//mat3[(i * this->m_dim_x) + j] += (mat1[(i << n) + k] * mat2[j + (k << n)]);

					//if (i % 5 == 0 && j % 5 == 0 && k % 5 == 0)
						//cout << "(i, j, k) = (" << i << ", " << j << ", " << k << ")" << endl;

					//sum += 
					//sum += (mat1[i * N + k] * mat2[j + k * N]);
					//sum += (mat1[A_index] * mat2[B_index]);
				}
				//C_index = (i * N + j);
				//mat3[C_index] = sum;

			}
		}
		
		for (unsigned int lhs_row = 0; lhs_row < this->m_dim_x; lhs_row++)
		{
			for (unsigned int rhs_col = 0; rhs_col < mat.m_dim_y; rhs_col++)
			{
				for (unsigned int rhs_row = 0; rhs_row < mat.m_dim_x; rhs_row++)
				{
					new_mat_elem += ((this->m_matrix[lhs_row][rhs_row]) * (mat.m_matrix[rhs_row][rhs_col]));
				}

				new_mat[lhs_row][rhs_col] = new_mat_elem;
				new_mat_elem = 0.0;
			}
		}*/

		this->m_dim_x = this->m_dim_x;
		this->m_dim_y = mat.m_dim_y;
		this->m_matrix = new_mat;
		//this->i_calc_trace();
		//this->i_calc_determinant();
	}

	return *this;
	
}

// Multiplication (C = A * B)
const Operator Operator::operator*(const Operator& mat) const
{
	Operator mat_intermediate = *this;
	mat_intermediate *= mat;

	return mat_intermediate;
}

/* ************************************* OPERATORS ************************************** */

// A^T
void Operator::transpose()
{
	unsigned int new_dim_x = m_dim_y;
	unsigned int new_dim_y = m_dim_x;

	complex<double> temp;

	// If matrix is square, perform the easier swap
	if (m_dim_x == m_dim_y)
	{
		for (unsigned int row = 0; row < m_dim_x - 1; row++)
		{
			for (unsigned int col = row + 1; col < m_dim_y; col++)
			{
				temp = m_matrix[row][col];
				m_matrix[row][col] = m_matrix[col][row];
				m_matrix[col][row] = temp;
			}
		}
	}

	else
	{
		vector<vector<complex<double>>> new_mat(new_dim_x, vector<complex<double>>(new_dim_y, 0.0));
		for (unsigned int row = 0; row < m_dim_x; row++)
		{
			for (unsigned int col = 0; col < m_dim_y; col++)
			{
				new_mat[col][row] = m_matrix[row][col];
			}
		}
		m_matrix = new_mat;
	}

	m_dim_x = new_dim_x;
	m_dim_y = new_dim_y;
}

// A*
void Operator::conjugate()
{
	vector<vector<complex<double>>>::iterator row_it;
	vector<complex<double>>::iterator col_it;

	for (row_it = m_matrix.begin(); row_it != m_matrix.end(); row_it++)
	{
		for (col_it = row_it->begin(); col_it != row_it->end(); col_it++)
		{
			*col_it = std::conj(*col_it);
		}
	}

	i_calc_trace();
}

// Hermitian congugate == Congugate transpose (A*^T)
void Operator::hermitian_congugate()
{
	conjugate();
	transpose();
}



// Calculate the easy cases - 2 x 2, 3 x 3
// TODO: Determine method for calculating determinant for n x n matrices - maybe just use det(A) = PI(lambda_i)?
void Operator::i_calc_determinant()
{
	if (m_dim_x == 2 && m_dim_y == 2)
	{
		m_determinant = ((m_matrix.at(0).at(0) * m_matrix.at(1).at(1)) - (m_matrix.at(0).at(1) * m_matrix.at(1).at(0)));
	}
	else
	{
		m_determinant = 0.0;
	}
	
}

/*vector<vector<complex<double>>>::const_iterator Operator::get_start_address() const
{
	return m_matrix.begin();
}
vector<vector<complex<double>>>::const_iterator Operator::get_end_address() const
{
	return m_matrix.end();
}*/

Operator Operator::get_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2)
{
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	//E.inverse();
	vector<vector<complex<double>>> sub_matrix_elements;// (row2 - row1 + 1, vector<complex<double>>(col2 - col1 + 1, 0.0));
	vector<complex<double>> submat_row;

	for (vector<vector<complex<double>>>::iterator row_it = m_matrix.begin() + row1; row_it <= m_matrix.begin() + row2; row_it++)
	{
		submat_row.assign(row_it->begin() + col1, row_it->begin() + col2 + 1);
		sub_matrix_elements.push_back(submat_row);
		submat_row.clear();
	}

	Operator submat(sub_matrix_elements);

	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

	cout << "--> GET TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

	return submat;
}

void Operator::set_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2, const vector<vector<complex<double>>>& submatrix)
{
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	int i = 0;
	for (vector<vector<complex<double>>>::iterator row_it = m_matrix.begin() + row1; row_it <= m_matrix.begin() + row2; row_it++)
	{
		std::copy(submatrix[i].begin(), submatrix[i].end(), row_it->begin() + col1);
		i++;
	}
	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

	cout << "--> SET TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
}

void Operator::inverse()
{
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	Operator m_inverse(m_dim_x, m_dim_y);

	m_inverse.createIdentityMatrix();

	unsigned int pivot_row = 0;
	unsigned int swap_row = 0;
	unsigned int pivot_col = 0;

	double pivot_value = 0.0;
	double element_norm = 0.0;
	complex<double> factor = 0.0;
	
	unsigned int ORIG_DIM_X = m_dim_x;
	unsigned int ORIG_DIM_Y = m_dim_y;
	unsigned int iteration = 0;

	unsigned int X_DIM_ADJ = m_dim_x - 1;
	unsigned int Y_DIM_ADJ = m_dim_y - 1;

	this->augment(m_inverse);

	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

	cout << "INIT TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

	start_time = std::chrono::steady_clock::now();
	// While still within the matrix
	while (pivot_row < X_DIM_ADJ && pivot_col < Y_DIM_ADJ)
	{
		// Get the element at the pivot
		pivot_value = std::norm(m_matrix[pivot_row][pivot_col]);
		swap_row = pivot_row;

		// For each row
		for (unsigned int i = pivot_row; i < ORIG_DIM_X; i++)
		{
			// If the element in the current row and pivot column is larger than the stored pivot value, update the pivot value and the index of the row to swap
			
			element_norm = std::norm(m_matrix[i][pivot_col]);
			
			if (element_norm > pivot_value)
			{
				pivot_value = element_norm;
				swap_row = i;
			}
		}

		// If pivot is non-zero, can perform the Gaussian elimination
		if (!iszero(pivot_value))
		{
			// Only need to swap if pivot row was not equal to swap row
			if (swap_row != pivot_row)
			{
				m_matrix[pivot_row].swap(m_matrix[swap_row]);
				//m_inverse.m_matrix[pivot_row].swap(m_inverse.m_matrix[swap_row]);
				//std::swap(m_matrix[pivot_row], m_matrix[swap_row]);
			}

			// Perform the Gaussian Elimination on the remaining elements in the matrix
			for (unsigned int i = pivot_row + 1; i < ORIG_DIM_X; i++)
			{
				// Get the pivot factor
				factor = m_matrix[i][pivot_col] / m_matrix[pivot_row][pivot_col];

				// Update each element
				for (unsigned int j = pivot_col; j < m_dim_y; j++)
				{
					m_matrix[i][j] -= (m_matrix[pivot_row][j] * factor);
					//m_matrix[i][j] -= (m_matrix[pivot_row][j] * (m_matrix[i][pivot_col] / m_matrix[pivot_row][pivot_col]));
				}
			}

			pivot_row++;
			//pivot_col++;
		}
		pivot_col++;
		// If the elements in the pivot column were all 0, skip this column and move to next one
		//else
		//{
			//pivot_col++;
		//}
		//this->print();
	}
	stop_time = std::chrono::steady_clock::now();

	cout << "ROW ECHELON TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

	start_time = std::chrono::steady_clock::now();
	// At ths point the matrix has been reduced to row echelon form - need to perform back substitution to eliminate off diagonal elements
	
	for (int j = ORIG_DIM_X - 1; j >= 1; j--)
	{
		for (int i = j - 1; i >= 0; i--)
		{
			if (!iszero(m_matrix[i][j]))
			{
				factor = m_matrix[i][j] / m_matrix[j][j];
				for (int k = j; k < m_dim_y; k++)
				{
					m_matrix[i][k] -= (m_matrix[j][k] * factor);
				}
			}
		}
	}

	stop_time = std::chrono::steady_clock::now();
	cout << "INVERSE TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

	start_time = std::chrono::steady_clock::now();
	// Normalize row
	for (int i = 0; i < m_dim_x; i++)
	{
		factor = 1.0 / m_matrix[i][i];
		m_matrix[i][i] = 1.0;
		std::transform(m_matrix[i].begin() + ORIG_DIM_Y, m_matrix[i].end(), m_matrix[i].begin() + ORIG_DIM_Y, std::bind1st(std::multiplies<complex<double>>(), factor));
	}
	stop_time = std::chrono::steady_clock::now();
	cout << "NORMALIZE TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
	
	for (int i = 0; i < m_dim_x; i++)
	{
		m_matrix[i].assign(m_matrix[i].begin() + ORIG_DIM_Y, m_matrix[i].end());
	}

	m_dim_x = m_matrix.size();
	m_dim_y = m_matrix[0].size();
}

void Operator::augment(Operator& A)
{
	if (this->m_dim_x == A.m_dim_x)
	{
		for (unsigned int i = 0; i < this->m_dim_x; i++)
		{
			this->m_matrix[i].insert(this->m_matrix[i].end(), A.m_matrix[i].begin(), A.m_matrix[i].end());
		}
	}

	this->m_dim_x = this->m_matrix.size();
	this->m_dim_y = this->m_matrix[0].size();
}