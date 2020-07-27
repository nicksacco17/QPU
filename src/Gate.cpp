
#include <iostream>
#include <iomanip>
#include "../include/Gate.h"

using std::cout;
using std::endl;
using namespace std::complex_literals;

#define PI 3.141592654

Gate::Gate()
{
	m_num_row = 2;
	m_num_col = 2;
	m_trace = 2;
	m_determinant = 1;
	m_name = "DEFAULT";

	m_mat = { 1, 0, 0, 1 };
}

Gate::~Gate()
{

}

Gate1Q::Gate1Q() : Gate()
{

}

Gate1Q::Gate1Q(const string gate_type)
{
	m_num_row = 2;
	m_num_col = 2;

	switch (gate_type[0])
	{
		case 'X':
			m_trace = 0.0;
			m_determinant = -1;
			m_name = "Pauli-X";

			m_mat = { 0, 1, 1, 0 };
			break;
		case 'Y':
			m_trace = 0.0;
			m_determinant = -1;
			m_name = "Pauli-Y";

			m_mat = { 0, -1i, 1i, 0 };
			break;
		case 'Z':
			m_trace = 0.0;
			m_determinant = -1;
			m_name = "Pauli-Z";

			m_mat = { 1, 0, 0, -1 };
			break;
		case 'H':
			m_trace = 0.0;
			m_determinant = -1.0 * std::sqrt(2);
			m_name = "Hadamard";

			m_mat = { (1.0 / std::sqrt(2)), (1.0 / std::sqrt(2)), (1.0 / std::sqrt(2)), (-1.0 / std::sqrt(2)) };
			break;
		case 'P':
			m_trace = (1.0 + 1.0i);
			m_determinant = 1i;
			m_name = "Phase (S/P)";

			m_mat = { 1, 0, 0, 1i };
			break;
		case 'T':
			m_trace = (1.0 + std::exp((0, PI / 4.0)));
			m_determinant = (1.0 + std::exp((0, PI / 4.0)));
			m_name = "T (PI/8)";

			m_mat = { 1, 0, 0,  std::exp((0, PI / 4.0)) };
			break;
		case 'S':
			m_trace = (1.0 + 1i);
			m_determinant = 9999;
			m_name = "Square Root NOT";

			m_mat = { (0.5) * (1.0 + 1i), (0.5) * (1.0 - 1i), (0.5) * (1.0 - 1i), (0.5) * (1.0 + 1i) };
			break;
		case '0':
			m_trace = 9999;
			m_determinant = 9999;
			m_name = "|0><0|";

			m_mat = { 1, 0, 0, 0 };
			break;
		case '1':
			m_trace = 9999;
			m_determinant = 9999;
			m_name = "|0><1|";

			m_mat = { 0, 1, 0, 0 };
			break;
		case '2':
			m_trace = 9999;
			m_determinant = 9999;
			m_name = "|1><0|";

			m_mat = { 0, 0, 1, 0 };
			break;
		case '3':
			m_trace = 9999;
			m_determinant = 9999;
			m_name = "|1><1|";

			m_mat = { 0, 0, 0, 1 };
			break;
		case 'I': default:
			m_trace = 2;
			m_determinant = 1;
			m_name = "I2";

			m_mat = { 1, 0, 0, 1 };
			break;
	}
}

void Gate1Q::set_gate(const string gate_type)
{
	switch (gate_type[0])
	{
		case 'X':
			m_trace = 0.0;
			m_determinant = -1;
			m_name = "Pauli-X";

			m_mat = { 0, 1, 1, 0 };
			break;
		case 'Y':
			m_trace = 0.0;
			m_determinant = -1;
			m_name = "Pauli-Y";

			m_mat = { 0, -1i, 1i, 0 };
			break;
		case 'Z':
			m_trace = 0.0;
			m_determinant = -1;
			m_name = "Pauli-Z";

			m_mat = { 1, 0, 0, -1 };
			break;
		case 'H':
			m_trace = 0.0;
			m_determinant = -1.0 * std::sqrt(2);
			m_name = "Hadamard";

			m_mat = { (1.0 / std::sqrt(2)), (1.0 / std::sqrt(2)), (1.0 / std::sqrt(2)), (-1.0 / std::sqrt(2)) };
			break;
		case 'P':
			m_trace = (1.0 + 1.0i);
			m_determinant = 1i;
			m_name = "Phase (S/P)";

			m_mat = { 1, 0, 0, 1i };
			break;
		case 'T':
			m_trace = (1.0 + std::exp((0, PI / 4.0)));
			m_determinant = (1.0 + std::exp((0, PI / 4.0)));
			m_name = "T (PI/8)";

			m_mat = { 1, 0, 0,  std::exp((0, PI / 4.0)) };
			break;
		case 'S':
			m_trace = (1.0 + 1i);
			m_determinant = 9999;
			m_name = "Square Root NOT";

			m_mat = { (0.5) * (1.0 + 1i), (0.5) * (1.0 - 1i), (0.5) * (1.0 - 1i), (0.5) * (1.0 + 1i) };
			break;
		case '0':
			m_trace = 9999;
			m_determinant = 9999;
			m_name = "|0><0|";

			m_mat = { 1, 0, 0, 0 };
			break;
		case '1':
			m_trace = 9999;
			m_determinant = 9999;
			m_name = "|0><1|";

			m_mat = { 0, 1, 0, 0 };
			break;
		case '2':
			m_trace = 9999;
			m_determinant = 9999;
			m_name = "|1><0|";

			m_mat = { 0, 0, 1, 0 };
			break;
		case '3':
			m_trace = 9999;
			m_determinant = 9999;
			m_name = "|1><1|";

			m_mat = { 0, 0, 0, 1 };
			break;
		case 'I': default:
			m_trace = 2;
			m_determinant = 1;
			m_name = "I2";

			m_mat = { 1, 0, 0, 1 };
			break;
	}
}

Gate1Q::Gate1Q(const string gate_type, complex<double> phase_arg)
{
	m_num_row = 2;
	m_num_col = 2;

	if (gate_type == "R")
	{
		m_trace = 9999;
		m_determinant = 9999;
		m_name = "Rotation";

		m_mat = { 1, 0, 0, std::exp((0, phase_arg)) };
	}
	else
	{
		m_trace = 2;
		m_determinant = 1;
		m_name = "I2";

		m_mat = { 1, 0, 0, 1 };
	}
}

Gate1Q::~Gate1Q()
{

}

Gate2Q::Gate2Q(const string gate_type)
{
	m_num_row = 4;
	m_num_col = 4;

	if (gate_type == "SWAP")
	{
		m_trace = 2;
		m_determinant = 9999;
		m_name = "SWAP";

		m_mat = 
		{ 
			1, 0, 0, 0, 
			0, 0, 1, 0, 
			0, 1, 0, 0, 
			0, 0, 0, 1 
		};
	}

	else if (gate_type == "CNOT")
	{
		m_trace = 9999;
		m_determinant = 9999;
		m_name = "Controlled-NOT";

		m_mat =
		{
			1, 0, 0, 0, 
			0, 1, 0, 0,
			0, 0, 0, 1,
			0, 0, 1, 0
		};
	}
	else if (gate_type == "SQRT-SWAP")
	{
		m_trace = 9999;
		m_determinant = 9999;
		m_name = "Square-Root SWAP";

		m_mat =
		{ 
			1, 0, 0, 0, 
			0, (0.5) * (1.0 + 1i), (0.5) * (1.0 - 1i), 0, 
			0, (0.5) * (1.0 - 1i), (0.5) * (1.0 + 1i), 0, 
			0, 0, 0, 1 
		};
	}
	else if (gate_type == "CU")
	{
		m_trace = 9999;
		m_determinant = 9999;
		m_name = "Controlled-U";

		m_mat = 
		{ 
			1, 0, 0, 0, 
			0, 1, 0, 0, 
			0, 0, 1, 0, 
			0, 0, 0, 1 
		};
	}
	else if (gate_type == "I")
	{
		m_trace = 2;
		m_determinant = 1;
		m_name = "I4";

		m_mat = 
		{ 
			1, 0, 0, 0, 
			0, 1, 0, 0, 
			0, 0, 1, 0, 
			0, 0, 0, 1 
		};
	}
	else
	{
		m_trace = 2;
		m_determinant = 1;
		m_name = "I4";

		m_mat = 
		{ 
			1, 0, 0, 0, 
			0, 1, 0, 0, 
			0, 0, 1, 0, 
			0, 0, 0, 1 
		};
	}

}

Gate2Q::~Gate2Q()
{

}

void Gate::print()
{
	cout << "---------- PRINT QUANTUM LOGIC GATE ----------" << endl;
	cout << "NAME: " << m_name << endl;
	cout << "DIMENSION: (" << m_num_row << " x " << m_num_col << ")" << endl;

	cout << "ELEMENTS:\n" << endl;

	for (unsigned int i = 0; i < m_num_row; i++)
	{
		cout << "| ";
		for (unsigned int j = 0; j < m_num_col; j++)
		{
			cout << std::setw(8) << std::setfill(' ') << std::setprecision(6) << std::fixed << m_mat.at(RC_TO_INDEX(i, j, m_num_col));
		}
		cout << "|" << endl;
	}

	cout << "---------- PRINT QUANTUM LOGIC GATE ----------" << endl;
}
