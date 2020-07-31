
#include "../include/QuantumDynamics.h"
#include "../include/LinearAlgebra.h"

#include <complex>

using namespace std::complex_literals;

QuantumDynamics::QuantumDynamics()
{
	m_H.set_dims(2, 2);
	m_H.createIdentityMatrix();

	//m_psi.set_dim(2);

	m_num_iterations = 10000;
	m_time_step = 1.0 / m_num_iterations;
	
	m_intermediate_states = vector<State>(m_num_iterations);
}

QuantumDynamics::QuantumDynamics(Operator& H, State& PSI_0, double time_step)
{
	m_H = H;
	m_psi = PSI_0;

	m_time_step = time_step;

	m_num_iterations = (unsigned int)(1.0 / m_time_step);
	m_intermediate_states = vector<State>(m_num_iterations);
}

QuantumDynamics::~QuantumDynamics()
{
	m_H.~Operator();
	m_psi.~State();

	m_intermediate_states.clear();
}

void QuantumDynamics::evolve(unsigned int num_iterations)
{
	Operator H_t = m_H;
	m_intermediate_states.at(0) = m_psi;
	for (unsigned int k = 1; k < m_num_iterations; k++)
	{
		H_t *= (m_time_step * k * -1.0i);
		op_RHS(m_intermediate_states.at(k), H_t, m_intermediate_states.at(k - 1));
	}
	m_psi = m_intermediate_states.back();
}
