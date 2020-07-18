#pragma once

#include "Operator.h"

class Pauli_Matrix : public Operator
{

public:

	Pauli_Matrix(string pauli_type);

	~Pauli_Matrix();

	bool is_square();

	bool is_Hermitian();

	bool is_Unitary();

	void print();

private:
	string m_name;
};