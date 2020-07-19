
#include <iostream>
#include <iomanip>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <functional>
#include "../include/Operator.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;
using std::string;

Operator::Operator()
{
	m_num_row = 2;
	m_num_col = 2;
	m_trace = 0.0;
	m_determinant = 0.0;

	m_mat = { {1, 0}, {0, 1} };
	m_determinant = 1;
	m_trace = 2;

	m_eigenvalues = {1, 1};
	m_eigenvectors = {{1, 0}, {0, 1}};
}

Operator::Operator(unsigned int in_row, unsigned int in_col)
{
	m_num_row = in_row;
	m_num_col = in_col;

	m_dim = (m_num_row == m_num_col) ? m_num_row : -1;
	
	m_mat = vector<complex<double>>(m_num_row * m_num_col, 0.0);

	m_determinant = 9999;
	m_trace = 9999;
}

Operator::Operator(const vector<vector<complex<double>>>& in_mat)
{
	m_num_row = in_mat.size();
	m_num_col = in_mat.at(0).size();

	m_dim = (m_num_row == m_num_col) ? m_num_row : -1;

	for (unsigned int i = 0; i < in_mat.size(); i++)
	{
		m_mat.insert(m_mat.end(), in_mat.at(i).begin(), in_mat.at(i).end());
	}

	m_determinant = 9999;
	m_trace = 9999;
}

Operator::Operator(const vector<complex<double>>& in_vec, unsigned int in_row, unsigned int in_col)
{
	m_mat = in_vec;

	m_num_row = in_row;
	m_num_col = in_col;
}

Operator::~Operator()
{
	m_mat.clear();
	m_eigenvalues.clear();
	m_eigenvectors.clear();
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
			cout << m_mat.at(RC_TO_INDEX(i, j)) << " ";
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
			if (iszero_print(m_mat.at(RC_TO_INDEX(i, j))))
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