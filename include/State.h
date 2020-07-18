
/*
 * *****************************************************************************
 *
 *	Project: QPU (Quantum Processing Unit)
 *	Class: State
 *	Description: State representation for N-qubit quantum system
 *
 *	Author: N. SACCO
 *	Date Updated: JUL/06/2020
 *
 *	VERSION HISTORY ------------------------------------------------------------
 *
 *	<V0.0.0>	(JUL/06/20)
 *	Description: Initial release w/base functionality
 *
 *	----------------------------------------------------------------------------
 *
 * *****************************************************************************
*/

#ifndef STATE_H
#define STATE_H

#include <complex>
#include <vector>
#include <string>

using std::string;
using std::vector;
using std::complex;

/* ********************************************************************************************** */

class State
{
public:

	/* ************************************ CONSTRUCTORS ************************************ */

	// Default Constructor
	State();

	// Parameterized Constructors

	// State (empty) w/size N
	State(unsigned int in_size);
	// State (full) w/contents and size N
	State(vector<complex<double>> in_vector);

	// Copy Constructor
	State(const State& psi);

	State(string state_type);

	// Destructor
	~State();

	/* ************************************ CONSTRUCTORS ************************************ */

	/* ************************************* OPERATORS ************************************** */

	State& operator=(const State& psi);					// Assignment (PSI_A = PSI_B)

	bool operator==(const State& psi);					// (Strict) Equality (PSI_A == PSI_B)
	
	bool operator!=(const State& psi);					// Not Equal (PSI_A != PSI_B)

	State& operator+=(const State& psi);				// Addition (PSI_A += PSI_B)

	const State operator+(const State& psi) const;		// Addition (PSI_C = PSI_A + PSI_B)

	State& operator-=(const State& psi);				// Subtraction (PSI_A -= PSI_B)

	const State operator-(const State& psi) const;		// Subtraction (PSI_C = PSI_A - PSI_B)

	State& operator*=(const complex<double> a);			// Multiplication (PSI_A *= a)

	const State operator*(const complex<double> a) const;		// Multiplication (PSI_B = PSI_A * a)

	/* ************************************* OPERATORS ************************************** */

	/* ************************************* FUNCTIONS ************************************** */

	// Getters

	// Get the size of the state
	unsigned int get_dim() const;

	// Get the magnitude of the state
	double get_magnitude() const;

	// Get the state element at spcified index
	complex<double> get_element(int index) const;

	vector<complex<double>> get_vector() const;

	// Populate the contents of the state with an array of elements
	void populate(const vector<complex<double>>& in_vector);

	// Determine if two states are equivalent up to a global phase shift (equivalent rays)
	bool approx(const State& psi);

	// Normalize the vector
	void normalize();

	// Print the contents of the state
	void print();

	void measure();

	vector<complex<double>>::const_iterator get_start_address() const;
	vector<complex<double>>::const_iterator get_end_address() const;

	/* ************************************* FUNCTIONS ************************************** */

private:

	void i_calc_magnitude();

	double m_magnitude;
	double m_norm_squared;
	unsigned int m_dim;
	vector<complex<double>> m_vector;
};

void entangle(State& entangled_state, State& psi_0, State& psi_1);

#endif // STATE_H