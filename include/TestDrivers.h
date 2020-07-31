
#ifndef TEST_DRIVERS_H
#define TEST_DRIVERS_H

// Test the State Class

//	TEST PORTION------------------------------------------STATUS

/*
	1. Constructor Test															
	2. Operators Test
		2A. CPU Operators
		2B. GPU Operators
	3. Mutators/Accessors Test
	4. Normalization Test
	5. Measurement Test
		5A. Probability Test
		5B: Collapse Test
	6. Special States
*/

void state_test_driver();

/*
	1. Constructor Test
	2. Operators Test
		2A. CPU Operators
		2B. GPU Operators
	3. Mutators/Accessors Test
	4. Normalization Test
	5. Measurement Test
		5A. Probability Test
		5B: Collapse Test
	6. Special States
*/

void matrix_test_driver();

void operator_test_driver();


void state_test_driver();

// Test the Operator Class
void operator_test_driver();

// Test Pauli Matrices
void pauli_test_driver();

// Test the Linear Algebra Class
void la_test_driver();

#endif // TEST_DRIVERS_H