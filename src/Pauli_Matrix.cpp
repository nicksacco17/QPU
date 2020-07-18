
#include <iostream>
#include <iomanip>
#include "Pauli_Matrix.h"

using std::cout;
using std::endl;
using namespace std::complex_literals;

Pauli_Matrix::Pauli_Matrix(string pauli_type)
{
	// Pauli-X Matrix
	// | 0 1 |
	// | 1 0 |
	if (pauli_type == "X" || pauli_type == "x")
	{
		m_dim_x = 2;
		m_dim_y = 2;
		m_trace = 0.0;
		m_determinant = -1;
		m_name = "Pauli-X";

		m_matrix = { {0, 1}, {1, 0} };
	}

	// Pauli-Y Matrix
	// | 0 -1i |
	// | 1i 0  |
	else if (pauli_type == "Y" || pauli_type == "y")
	{
		m_dim_x = 2;
		m_dim_y = 2;
		m_trace = 0.0;
		m_determinant = -1;
		m_name = "Pauli-Y";

		m_matrix = { {0, -1i}, {1i, 0} };
	}

	// Pauli-Z Matrix
	// | 1 0  |
	// | 0 -1 |
	else if (pauli_type == "Z" || pauli_type == "z")
	{
		m_dim_x = 2;
		m_dim_y = 2;
		m_trace = 0.0;
		m_determinant = -1;
		m_name = "Pauli-Z";

		m_matrix = { {1, 0}, {0, -1} };
	}

	// Identity-N matrix (N x N)
	// EX: I4 = | 1 0 0 0 |
	//			| 0 1 0 0 |
	//			| 0 0 1 0 |
	//			| 0 0 0 1 |
	else if (pauli_type.at(0) == 'I' || pauli_type.at(0) == 'i')
	{
		// Convert the remaining part of the input string to an integer and use as dimenson of Identity
		int size = stoi(pauli_type.substr(1));

		m_dim_x = size;
		m_dim_y = size;
		m_trace = size;
		m_determinant = 1;
		m_name = "Identity-" + pauli_type.substr(1);

		m_matrix.clear();
		for (unsigned int i = 0; i < m_dim_x; i++)
		{
			vector<complex<double>> row(m_dim_y, 0.0);
			row.at(i) = 1.0;
			m_matrix.push_back(row);
		}
	}
	// Else invalid Pauli matrix, so just create I2
	else
	{
		m_dim_x = 2;
		m_dim_y = 2;
		m_trace = 2;

		m_matrix = { {1, 0}, {1, 0} };
	}
}

Pauli_Matrix::~Pauli_Matrix()
{
	m_matrix.clear();
}

bool Pauli_Matrix::is_square()
{
	return true;
}

bool Pauli_Matrix::is_Hermitian()
{
	return true;
}

bool Pauli_Matrix::is_Unitary()
{
	return true;
}

void Pauli_Matrix::print()
{
	cout << "---------- PRINT PAULI MATRIX ----------" << endl;

	cout << "NAME: " << m_name << endl;
	cout << "DIMENSION: (" << m_dim_x << " x " << m_dim_y << ")" << endl;
	cout << "TRACE: " << m_trace << endl;
	cout << "DETERMINANT: " << m_determinant << endl;

	cout << "SQUARE: " << std::boolalpha << is_square() << endl;
	cout << "HERMITIAN: " << std::boolalpha << is_Hermitian() << endl;
	cout << "UNITARY: " << std::boolalpha << is_Unitary() << endl;
	cout << "ELEMENTS:\n" << endl;

	vector<vector<complex<double>>>::const_iterator row_it;
	vector<complex<double>>::const_iterator col_it;

	for (row_it = m_matrix.begin(); row_it != m_matrix.end(); row_it++)
	{
		cout << "| ";
		for (col_it = row_it->begin(); col_it != row_it->end(); col_it++)
		{
			cout << std::setw(8) << std::setfill(' ') << std::setprecision(6) << std::fixed << *col_it;
		}
		cout << "|" << endl;
	}

	cout << "\n---------- PRINT PAULI MATRIX ----------" << endl;
}