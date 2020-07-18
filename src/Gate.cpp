
#include <iostream>
#include <iomanip>
#include "../include/Gate.h"

using std::cout;
using std::endl;
using namespace std::complex_literals;

#define PI 3.141592654

Gate::Gate()
{
	m_dim_x = 2;
	m_dim_y = 2;
	m_trace = 2;
	m_determinant = 1;
	m_name = "DEFAULT";

	m_matrix = { {1, 0}, {0, 1} };
}

Gate::~Gate()
{

}


Gate1Q::Gate1Q(string gate_type)
{
	m_dim_x = 2;
	m_dim_y = 2;

	if (gate_type == "X")
	{
		m_trace = 0.0;
		m_determinant = -1;
		m_name = "Pauli-X";

		m_matrix = { {0, 1}, {1, 0} };
	}
	else if (gate_type == "Y")
	{
		m_trace = 0.0;
		m_determinant = -1;
		m_name = "Pauli-Y";

		m_matrix = { {0, -1i}, {1i, 0} };
	}
	else if (gate_type == "Z")
	{
		m_trace = 0.0;
		m_determinant = -1;
		m_name = "Pauli-Z";

		m_matrix = { {1, 0}, {0, -1} };
	}
	else if (gate_type == "H")
	{
		m_trace = 0.0;
		m_determinant = -1.0 * std::sqrt(2);
		m_name = "Hadamard";

		m_matrix = { {(1.0 / std::sqrt(2)), (1.0 / std::sqrt(2))}, {(1.0 / std::sqrt(2)), (-1.0 / std::sqrt(2))} };
	}
	else if (gate_type == "P")
	{
		m_trace = (1.0 + 1.0i);
		m_determinant = 1i;
		m_name = "Phase (S/P)";

		m_matrix = { {1, 0}, {0, 1i} };
	}
	else if (gate_type == "T")
	{
		m_trace = (1.0 + std::exp(1i * PI / 4.0));
		m_determinant = (1.0 + std::exp(1i * PI / 4.0));
		m_name = "T (PI/8)";
		
		m_matrix = { {1, 0}, {0,  std::exp(1i * PI / 4.0)} };
	}
	else if (gate_type == "SQRT-NOT")
	{
		m_trace = (1.0 + 1i);
		m_determinant = 9999;
		m_name = "Square Root NOT";

		m_matrix = { {(0.5) * (1.0 + 1i), (0.5) * (1.0 - 1i)}, {(0.5) * (1.0 - 1i), (0.5) * (1.0 + 1i)} };
	}
	else if (gate_type == "I")
	{
		m_trace = 2;
		m_determinant = 1;
		m_name = "I2";

		m_matrix = { {1, 0}, {0, 1} };
	}
	else
	{
		m_trace = 2;
		m_determinant = 1;
		m_name = "I2";

		m_matrix = { {1, 0}, {0, 1} };
	}
}

Gate1Q::Gate1Q(string gate_type, complex<double> phase_arg)
{
	m_dim_x = 2;
	m_dim_y = 2;

	if (gate_type == "R")
	{
		m_trace = 9999;
		m_determinant = 9999;
		m_name = "Rotation";

		m_matrix = { {1, 0}, {0, std::exp(1i * phase_arg)} };
	}
	else
	{
		m_trace = 2;
		m_determinant = 1;
		m_name = "I2";

		m_matrix = { {1, 0}, {0, 1} };
	}
}

Gate1Q::~Gate1Q()
{

}

Gate2Q::Gate2Q(string gate_type)
{
	m_dim_x = 4;
	m_dim_y = 4;

	if (gate_type == "SWAP")
	{
		m_trace = 2;
		m_determinant = 9999;
		m_name = "SWAP";

		m_matrix = 
		{ 
			{ 1, 0, 0, 0 }, 
			{ 0, 0, 1, 0 }, 
			{ 0, 1, 0, 0 }, 
			{ 0, 0, 0, 1 } 
		};
	}

	else if (gate_type == "CNOT")
	{
		m_trace = 9999;
		m_determinant = 9999;
		m_name = "Controlled-NOT";

		m_matrix =
		{
			{ 1, 0, 0, 0 }, 
			{ 0, 1, 0, 0 },
			{ 0, 0, 0, 1 },
			{ 0, 0, 1, 0 }
		};
	}
	else if (gate_type == "SQRT-SWAP")
	{
		m_trace = 9999;
		m_determinant = 9999;
		m_name = "Square-Root SWAP";

		m_matrix =
		{ 
			{ 1, 0, 0, 0 }, 
			{ 0, (0.5) * (1.0 + 1i), (0.5) * (1.0 - 1i), 0 }, 
			{ 0, (0.5) * (1.0 - 1i), (0.5) * (1.0 + 1i), 0 }, 
			{ 0, 0, 0, 1 } 
		};
	}
	else if (gate_type == "CU")
	{
		m_trace = 9999;
		m_determinant = 9999;
		m_name = "Controlled-U";

		m_matrix = { {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
	}
	else if (gate_type == "I")
	{
		m_trace = 2;
		m_determinant = 1;
		m_name = "I4";

		m_matrix = { {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
	}
	else
	{
		m_trace = 2;
		m_determinant = 1;
		m_name = "I4";

		m_matrix = { {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
	}

}

Gate2Q::~Gate2Q()
{

}

void Gate::print()
{
	cout << "---------- PRINT QUANTUM LOGIC GATE ----------" << endl;
	cout << "NAME: " << m_name << endl;
	cout << "DIMENSION: (" << m_dim_x << " x " << m_dim_y << ")" << endl;

	cout << "ELEMENTS:\n" << endl;

	for (int i = 0; i < m_dim_x; i++)
	{
		cout << "| ";
		for (int j = 0; j < m_dim_y; j++)
		{
			cout << std::setw(8) << std::setfill(' ') << std::setprecision(6) << std::fixed << m_matrix[i][j];
		}
		cout << "|" << endl;
	}

	cout << "---------- PRINT QUANTUM LOGIC GATE ----------" << endl;
}