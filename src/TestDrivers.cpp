
#include <iostream>
#include <ctime>
#include <cmath>	

#include "State.h"
#include "Operator.h"
#include "LinearAlgebra.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;

#define NUM_QUBITS 3
#define STATE_SIZE unsigned int(std::pow(2, NUM_QUBITS))

void state_test_driver()
{
	cout << "STATE TEST DRIVER" << endl;

	std::vector<std::complex<double>> test_vector(STATE_SIZE);

	for (std::vector<std::complex<double>>::iterator i = test_vector.begin(); i != test_vector.end(); i++)
	{
		*i = 1;
	}

	std::time_t start_time = std::time(NULL);
	cout << "Creating state..." << endl;

	State* my_state = new State(STATE_SIZE);
	State* my_state2 = new State();
	State* my_state3 = new State(STATE_SIZE);

	cout << "Populating state..." << endl;
	my_state->populate(test_vector);
	my_state3->populate(test_vector);
	my_state->print();
	my_state3->print();

	*my_state3 += *my_state;

	cout << "SUM!" << endl;
	my_state3->print();


	cout << "Normalizing state..." << endl;
	my_state->normalize();
	my_state->print();

	*my_state2 = *my_state;

	cout << "STATE 2" << endl;
	my_state2->print();

	std::time_t stop_time = std::time(NULL);
	cout << "DONE!" << endl;

	std::time_t total_time = stop_time - start_time;

	cout << "TOTAL TIME ELAPSED: " << total_time << " SECONDS!" << endl;

	delete my_state;
	delete my_state2;
}

void operator_test_driver()
{
	/*
	cout << "OPERATOR TEST DRIVER" << endl;

	vector<vector<complex<double>>> test_matrix(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));
	for (int i = 0; i < STATE_SIZE; i++)
	{
		for (int j = 0; j < STATE_SIZE; j++)
		{
			test_matrix.at(i).at(j) = i + j;
		}
	}

	Operator test_operator_default();
	Operator test_operator_empty_N(STATE_SIZE, STATE_SIZE);
	Operator test_operator_fill_N(test_matrix);
	Operator pauli_x();
	Operator pauli_y();
	Operator pauli_z();

	Operator* my_operator = new Operator();

	my_operator->print();

	std::vector<std::vector<std::complex<double>>> test_vector;

	for (unsigned int i = 0; i < STATE_SIZE; i++)
	{
		test_vector.push_back(std::vector<std::complex<double>>(STATE_SIZE, i));
	}

	my_operator->populate(test_vector);

	my_operator->print();

	delete my_operator;
	*/
}

void pauli_test_driver()
{
	/*
	cout << "PAULI TEST DRIVER" << endl;

	Operator* pauli_x = new Operator("X", 2);
	Operator* pauli_y = new Operator("Y", 2);
	Operator* pauli_z = new Operator("Z", 2);
	Operator* identity2 = new Operator("I", 2);
	Operator* identity4 = new Operator("I", 4);

	cout << "PAULI X" << endl;
	pauli_x->print();

	cout << "PAULI Y" << endl;
	pauli_y->print();

	cout << "PAULI Z" << endl;
	pauli_z->print();

	cout << "IDENTITY 2" << endl;
	identity2->print();

	cout << "IDENTITY 4" << endl;
	identity4->print();

	delete pauli_x;
	delete pauli_y;
	delete pauli_z;
	delete identity2;
	delete identity4;
	*/
}

void la_test_driver()
{
	/*
	State* psi1 = new State(2);
	State* psi2 = new State(2);
	Operator* I2 = new Operator("I", 2);

	std::vector<std::complex<double>> psi1_vec = { 1, 1 };
	std::vector<std::complex<double>> psi2_vec = { 1, 1 };

	psi1->populate(psi1_vec);
	psi2->populate(psi2_vec);

	std::complex<double> m_dot = inner_product(*psi1, *psi2);

	State RHS = op_rhs(*I2, *psi1);

	psi1->print();
	psi2->print();
	
	cout << "MAT MUL" << endl;
	RHS.print();

	cout << "DOT PRODUCT = " << m_dot << endl;

	delete psi1;
	delete psi2;
	*/
}