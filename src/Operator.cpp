
#include <iostream>
#include <iomanip>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <functional>
#include <random>
#include <typeinfo>
#include "../include/Operator.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;
using std::string;

Operator::Operator() : Matrix()
{
	/*m_num_row = 2;
	m_num_col = 2;
	m_trace = 0.0;
	m_determinant = 0.0;

	m_mat = { {1, 0}, {0, 1} };
	m_determinant = 1;
	m_trace = 2;

	m_eigenvalues = {1, 1};
	m_eigenvectors = {{1, 0}, {0, 1}};
	*/
}

Operator::Operator(unsigned int in_row, unsigned int in_col) : Matrix(in_row, in_col)
{
	/*m_num_row = in_row;
	m_num_col = in_col;

	m_dim = (m_num_row == m_num_col) ? m_num_row : -1;
	
	m_mat = vector<complex<double>>(m_num_row * m_num_col, 0.0);

	m_determinant = 9999;
	m_trace = 9999;
	*/
}

Operator::Operator(const vector<vector<complex<double>>>& in_mat) : Matrix(in_mat)
{
	//Matrix(in_mat);

	/*m_num_row = in_mat.size();
	m_num_col = in_mat.at(0).size();

	m_dim = (m_num_row == m_num_col) ? m_num_row : -1;

	for (unsigned int i = 0; i < in_mat.size(); i++)
	{
		m_mat.insert(m_mat.end(), in_mat.at(i).begin(), in_mat.at(i).end());
	}

	m_determinant = 9999;
	m_trace = 9999;
	*/
}

Operator::Operator(const vector<complex<double>>& in_vec, unsigned int in_row, unsigned int in_col) : Matrix(in_vec, in_row, in_col)
{
	/*m_mat = in_vec;

	m_num_row = in_row;
	m_num_col = in_col;
	*/
}

Operator::Operator(string mat_type, unsigned int in_dim, double lower_range, double upper_range, long unsigned int seed) : Matrix(in_dim, in_dim)
{
#ifdef bhjsfhsdf
	cout << "I AM ON THE GPU" << endl;
#else
	m_dim = in_dim;
	m_num_row = m_dim;
	m_num_col = m_dim;

	// Reset matrix
	m_mat = vector<complex<double>>(m_num_row * m_num_col, 0.0);

	double real_part = 0.0;
	double imag_part = 0.0;

	complex<double> rand_num = 0.0;

	std::default_random_engine RAND_NUM_GENERATOR{ static_cast<long unsigned int>(seed) };

	// Real Int
	if (mat_type == "R-I")
	{
		std::uniform_int_distribution<int> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					rand_num = complex<double>(real_part, 0.0);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = rand_num;
				}
			}
		}
	}
	// Real Float
	else if (mat_type == "R-F")
	{
		std::uniform_real_distribution<float> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					rand_num = complex<double>(real_part, 0.0);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = rand_num;
				}
			}
		}
	}
	// Real Double
	else if (mat_type == "R-D")
	{
		std::uniform_real_distribution<double> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					rand_num = complex<double>(real_part, 0.0);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = rand_num;
				}
			}
		}
	}

	// Complex Int
	else if (mat_type == "C-I")
	{
		std::uniform_int_distribution<int> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					imag_part = distribution(RAND_NUM_GENERATOR);
					rand_num = complex<double>(real_part, imag_part);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = std::conj(rand_num);
				}
			}
		}
	}
	// Complex Float
	else if (mat_type == "C-F")
	{
		std::uniform_real_distribution<float> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					imag_part = distribution(RAND_NUM_GENERATOR);
					rand_num = complex<double>(real_part, imag_part);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = std::conj(rand_num);
				}
			}
		}
	}
	// Complex Double
	else if (mat_type == "C-D")
	{
		std::uniform_real_distribution<double> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					imag_part = distribution(RAND_NUM_GENERATOR);
					rand_num = complex<double>(real_part, imag_part);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = std::conj(rand_num);
				}
			}
		}
	}
	else
	{
		std::uniform_int_distribution<int> distribution(0, 100);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					rand_num = complex<double>(real_part, 0.0);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = rand_num;
				}
			}
		}
	}
#endif
}

Operator::~Operator()
{
	m_mat.clear();
	m_eigenvalues.clear();
	m_eigenvectors.clear();
}

Operator& Operator::operator=(const Matrix& mat)
{
	if (this != &mat)
	{
		this->m_num_row = mat.get_num_rows();
		this->m_num_col = mat.get_num_cols();
		this->m_mat = mat.get_row_order_mat();
	}

	return *this;
}

Operator Operator::get_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2)
{
	Operator SUB_MAT;

	unsigned int NUM_ROWS = (row2 - row1) + 1;
	unsigned int NUM_COLS = (col2 - col1) + 1;

	vector<complex<double>> sub_mat_elements;

	for (unsigned int i = row1; i <= row2; i++)
	{
		for (unsigned int j = col1; j <= col2; j++)
		{
			sub_mat_elements.push_back(this->m_mat[RC_TO_INDEX(i, j, m_num_col)]);
		}
	}

	SUB_MAT.set_matrix(sub_mat_elements, NUM_ROWS, NUM_COLS);
	return SUB_MAT;
}

void Operator::print() const
{
	cout << "---------- PRINT OPERATOR ----------" << endl;

	cout << "DIMENSION: (" << m_num_row << " x " << m_num_col << ")" << endl;
	cout << "TRACE: " << m_trace << endl;
	cout << "DETERMINANT: " << m_determinant << endl;

	cout << "ELEMENTS:\n" << endl;

	for (unsigned int i = 0; i < m_num_row; i++)
	{
		cout << "| ";
		for (unsigned int j = 0; j < m_num_col; j++)
		{
			//cout << "(i, j) = (" << i << ", " << j << ") = " << RC_TO_INDEX(i, j, m_num_col) << endl;
			cout << std::fixed << std::setprecision(6) << m_mat.at(RC_TO_INDEX(i, j, m_num_col)) << " ";
		}
		cout << "|" << endl;
	}

	cout << "\n---------- PRINT OPERATOR ----------" << endl;
}

void Operator::print_shape() const
{
	for (unsigned int i = 0; i < m_num_row; i++)
	{
		cout << "| ";
		for (unsigned int j = 0; j < m_num_col; j++)
		{
			if (iszero_print(m_mat.at(RC_TO_INDEX(i, j, m_num_col))))
			{
				cout << "0 ";
			}

			else if (i == j)
			{
				cout << "+ ";
			}

			else
			{
				cout << "* ";
			}
		}
		cout << "|" << endl;
	}
}
