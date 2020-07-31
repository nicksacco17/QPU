
#include <iostream>
#include <assert.h>
#include <cmath>
#include "../include/State.h"
#include "../include/Utility.h"

#include <algorithm>
#include <functional>
#include <string>
#include <iomanip>
#include <random>
#include <chrono>
#include <ctime>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::complex;

// Assume 2-level default
State::State()
{
	m_dim = 2;
	m_magnitude = 0;
	m_norm_squared = 0;
	m_vector = vector<complex<double>>(m_dim, 0.0);
}

// Empty N-level state
State::State(unsigned int in_size)
{
	m_dim = in_size;
	m_magnitude = 0;
	m_norm_squared = 0;
	m_vector = vector<complex<double>>(m_dim, 0.0);
}

// Populated N-level state
State::State(vector<complex<double>> in_vector)
{
	m_vector = in_vector;
	m_dim = m_vector.size();

	i_calc_magnitude();
}

// State (full) w/contents and size N
State::State(vector<complex<double>> in_vector, unsigned int in_size)
{
	m_vector = in_vector;
	m_dim = in_size;
}

// Copy Constructor
State::State(const State& psi)
{
	m_vector = psi.m_vector;
	m_dim = psi.m_dim;
	m_norm_squared = psi.m_norm_squared;
	m_magnitude = psi.m_magnitude;
}

// Deconstructor
State::~State()
{

}

State::State(string state_type)
{
	if (state_type == "PHI+")
	{
		m_dim = 4;
		m_vector = { INV_SQRT2, 0, 0, INV_SQRT2 };
		m_norm_squared = 9999;
		m_magnitude = 9999;
	}
	else if (state_type == "PHI-")
	{
		m_dim = 4;
		m_vector = { INV_SQRT2, 0, 0, -1.0 * INV_SQRT2 };
		m_norm_squared = 9999;
		m_magnitude = 9999;
	}
	else if (state_type == "PSI+")
	{
		m_dim = 4;
		m_vector = {0, INV_SQRT2, INV_SQRT2, 0};
		m_norm_squared = 9999;
		m_magnitude = 9999;
	}
	else if (state_type == "PSI-")
	{
		m_dim = 4;
		m_vector = {0, INV_SQRT2, -1.0 * INV_SQRT2, 0};
		m_norm_squared = 9999;
		m_magnitude = 9999;
	}
	else
	{
		m_dim = 4;
		m_vector = { 1, 1, 1, 1};
		m_norm_squared = 9999;
		m_magnitude = 9999;
	}
}

// Equality (PSI_A == PSI_B)
bool State::operator==(const State& psi)
{
	bool is_equal = true;

	// If the two states have the same dimensionality and the same magnitude,
	// then they are equal up to a phase shift.  Need to also check each element exactly.
	// TODO - Determine if it is better to define equality up to a phase shift, or total equality.
	if (this->m_dim == psi.get_dim())
	{
		vector<complex<double>>::const_iterator lhs_it = this->m_vector.begin();
		vector<complex<double>>::const_iterator rhs_it = psi.m_vector.begin();

		for (; is_equal, lhs_it != this->m_vector.end(), rhs_it != psi.m_vector.end(); lhs_it++, rhs_it++)
		{
			if (*lhs_it != *rhs_it)
			{
				is_equal = false;
			}
		}
	}
	else
	{
		is_equal = false;
	}

	return is_equal;
}


// Weak equality - the two states are equivalent up to a global phase shift
bool State::approx(const State& psi)
{
	bool is_equal = false;

	// If the two states have the same dimension and same magnitude, they are the same ray
	if (this->m_dim == psi.m_dim && this->m_magnitude == psi.m_magnitude)
	{
		is_equal = true;
	}

	return is_equal;
}

// Not Equal (PSI_A != PSI_B)
bool State::operator!=(const State& psi)
{
	bool is_equal = (*this == psi);
	return !is_equal;
}

// psi1 = psi2 <==> psi1.operator=(psi2) --> this == psi1
State& State::operator=(const State& psi)
{
	if (this != &psi)
	{
		this->m_dim = psi.m_dim;
		this->m_magnitude = psi.m_magnitude;
		this->m_norm_squared = psi.m_norm_squared;
		this->m_vector = psi.m_vector;
	}

	return *this;
}

State& State::operator+=(const State& psi)
{
	// If the calling object is different from the target object and they agree in dimension,
	// the two states can be added together

	if (this != &psi && this->m_dim == psi.m_dim)
	{
		vector<complex<double>>::iterator lhs_it = this->m_vector.begin();
		vector<complex<double>>::const_iterator rhs_it = psi.m_vector.begin();

		for (; lhs_it != this->m_vector.end(), rhs_it != psi.m_vector.end(); lhs_it++, rhs_it++)
		{
			*lhs_it += *rhs_it;
		}

		i_calc_magnitude();
	}

	return *this;
}

const State State::operator+(const State& psi) const
{
	State psi_intermediate = *this;
	psi_intermediate += psi;

	return psi_intermediate;
}

const State State::operator*(const complex<double> a) const
{
	State psi_intermediate = *this;
	psi_intermediate *= a;

	return psi_intermediate;
}

State& State::operator-=(const State& psi)
{
	if (this != &psi && this->m_dim == psi.m_dim)
	{
		vector<complex<double>>::iterator lhs_it = this->m_vector.begin();
		vector<complex<double>>::const_iterator rhs_it = psi.m_vector.begin();

		for (; lhs_it != this->m_vector.end(), rhs_it != psi.m_vector.end(); lhs_it++, rhs_it++)
		{
			*lhs_it -= *rhs_it;
		}

		i_calc_magnitude();
	}

	return *this;
}

const State State::operator-(const State& psi) const
{
	State psi_intermediate = *this;
	psi_intermediate -= psi;

	return psi_intermediate;
}

State& State::operator*=(const std::complex<double> a)
{
	vector<complex<double>>::iterator psi_it = this->m_vector.begin();

	for (; psi_it != this->m_vector.end(); psi_it++)
	{
		*psi_it *= a;
	}

	//i_calc_magnitude();
	
	return *this;
}


// Get the length of the state
unsigned int State::get_dim() const
{
	return m_dim;
}

complex<double> State::get_element(int index) const
{
	return m_vector.at(index);
}

void State::set_element(int index, complex<double> value)
{
	m_vector.at(index) = value;
}

vector<complex<double>> State::get_vector() const
{
	return m_vector;
}

// Get the magnitude of the state
double State::get_magnitude() const
{
	return m_magnitude;
}

void State::i_calc_magnitude()
{
	m_magnitude = 0;
	m_norm_squared = 0;

	for (vector<complex<double>>::iterator it = m_vector.begin(); it != m_vector.end(); it++)
	{
		m_norm_squared += std::norm(*it);
	}

	m_magnitude = std::sqrt(m_norm_squared);
}

// Populate the contents of the state with an array of elements
void State::populate(const vector<complex<double>>& in_vector)
{
	m_vector = in_vector;
	m_dim = m_vector.size();
	i_calc_magnitude();
}

// Normalize the state
void State::normalize()
{
	for (vector<complex<double>>::iterator it = m_vector.begin(); it != m_vector.end(); it++)
	{
		*it = *it / m_magnitude;
	}

	i_calc_magnitude();
}

// Print the contents of the state
void State::print()
{
	cout << "---------- PRINT VECTOR ----------" << endl;
	cout << "DIMENSION: " << m_dim << endl;
	cout << "MAGNITUDE: " << m_magnitude << endl;
	cout << "NORM SQUARED: " << m_norm_squared << endl;
	cout << "ELEMENTS: " << endl;

	for (vector<complex<double>>::iterator it = m_vector.begin(); it != m_vector.end(); it++)
	{
		cout << std::showpos << std::left << std::fixed << "| " << *it << "|" << endl;
	}

	cout << "---------- PRINT VECTOR ----------" << endl;
}

vector<complex<double>>::const_iterator State::get_start_address() const
{
	return m_vector.begin();
}

vector<complex<double>>::const_iterator State::get_end_address() const
{
	return m_vector.end();
}

/*
complex<double> State::inner_product(const State& psi)
{
	complex<double> m_inner_product = 0.0;

	vector<complex<double>>::const_iterator lhs_it = this->m_vector.begin();
	vector<complex<double>>::const_iterator rhs_it = psi.m_vector.begin();

	for (; lhs_it != this->m_vector.end(), rhs_it != psi.m_vector.end(); lhs_it++, rhs_it++)
	{
		m_inner_product += (std::conj(*lhs_it) * (*rhs_it));
	}

	return m_inner_product;
}
*/

void State::measure()
{
	int NUM_TEST = 100000000;
	//std::chrono::steady_clock::time_point time_seed = std::chrono::steady_clock::now();
	std::default_random_engine test_generator{ static_cast<long unsigned int>(time(0)) };// {time_seed};
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	vector<unsigned int> index_count(m_dim, 0);

	for (int counter = 0; counter < NUM_TEST; counter++)
	{
		if (counter % 100000 == 0)
		{
			cout << "ITERATION: " << counter << endl;
		}
		vector<double> cdf(m_dim, 0.0);

		cdf[0] = std::norm(m_vector[0]);

		for (unsigned int i = 1; i < m_dim; i++)
		{
			cdf[i] = std::norm(m_vector[i]) + cdf[i - 1];
		}

		double gen_num = distribution(test_generator);
		unsigned int selected_index = 0;

		//for (int k = 0; k < m_dim; k++)
		//{
			//cout << "cdf[" << k << "] = " << cdf[k] << endl;
		//}

		if (gen_num <= cdf[0])
		{
			selected_index = 0;
		}
		else
		{
			for (unsigned int j = 1; j < m_dim; j++)
			{
				if ((cdf[j - 1] < gen_num && gen_num <= cdf[j]))
				{
					selected_index = j;
				}
			}
		}
		index_count[selected_index]++;
	}

	for (unsigned int i = 0; i < m_dim; i++)
	{
		cout << "INDEX: " << i << " PROB = " << ((double)(index_count[i]) / NUM_TEST) << endl;
	}

	

	//cout << "GEN NUM = " << gen_num << endl;
	//cout << "SELECTED STATE = " << selected_index << endl;
	


	//mat1_rand.at(i).at(j) = distribution(test_generator);
}

void entangle(State& entangled_state, State& psi_0, State& psi_1)
{
	unsigned int DIM_PSI0 = psi_0.get_dim();
	unsigned int DIM_PSI1 = psi_1.get_dim();
	unsigned int entangled_state_dim = DIM_PSI0 * DIM_PSI1;
	
	vector<complex<double>> entangled_state_elements(entangled_state_dim, 0.0);

	for (unsigned int i = 0; i < DIM_PSI1; i++)
	{
		std::transform(psi_0.get_start_address(), psi_0.get_end_address(), entangled_state_elements.begin() + (DIM_PSI0 * i), std::bind1st(std::multiplies<complex<double>>(), psi_1.get_element(i)));
	}

	entangled_state.populate(entangled_state_elements);
}
