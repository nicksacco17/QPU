
#ifndef QUANTUM_CIRCUIT_H
#define QUANTUM_CIRCUIT_H

#include <string>
#include <initializer_list>
#include <vector>

#include "State.h"
#include "Operator.h"
#include "Gate.h"

using std::string;
using std::initializer_list;
using std::vector;

struct GateLevel
{
	unsigned int level;
	string gate_type;
	vector<unsigned int> qubit_list;
};

class QuantumCircuit
{
public:
	
	QuantumCircuit(unsigned int num_qubits);

	void add_gate(unsigned int level, string gate_type, initializer_list<unsigned int> qubits);

	// Print out contents of the circuit in text format
	void print();

	// Print out contents of the circuit in graphcal format
	void display();

	// Entangle particular qubits for specified entanglement
	void entangle(string entanglement_type, initializer_list<unsigned int> entangled_qubits);

	void initialize(initializer_list<unsigned int> qubits, State& init_state);

	void initialize(initializer_list<unsigned int> qubits, const string& state_type);

	void evolve();

	void sort_levels();

private:

	// Number of qubits in the circuit
	unsigned int m_num_qubits;

	unsigned int m_functional_size;

	// Gates in the circuit at a particular level

	// <GATE_INDEX, GATE_TYPE, QUBITS>

	vector<vector<GateLevel>> m_circuit_levels;
	
	Operator m_circuit_operator;

	State m_circuit_state;
};

#endif // QUANTUM_CIRCUIT_H