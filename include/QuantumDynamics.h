
#ifndef QUANTUM_DYNAMICS_H
#define QUANTUM_DYNAMICS_H

#include "State.h"
#include "Operator.h"

#include <vector>
#include <string>

using std::vector;
using std::string;

class QuantumDynamics
{
public:
	QuantumDynamics();

	QuantumDynamics(Operator& H, State& PSI_0, double time_step);

	~QuantumDynamics();

	void evolve(unsigned int num_iterations);

	void save(string file_dest);

	void print();

private:

	double m_time_step;
	unsigned int m_num_iterations;

	Operator m_H;
	State m_psi;

	vector<State> m_intermediate_states;

};


#endif