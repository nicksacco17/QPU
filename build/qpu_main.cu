/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include syevd_example.cpp 
 *   g++ -o a.out syevd_example.o -L/usr/local/cuda/lib64 -lcudart -lcusolver
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include "../include/Matrix.h"
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <iostream>
#include <string>

using std::vector;
using std::complex;
using std::cout;
using std::endl;

int main(int argc, char*argv[])
{
	#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
		cout << "RUNNING ON GPU" << endl;
	#else
		cout << "RUNNING ON CPU" << endl;
	#endif




	std::chrono::steady_clock::time_point total_start_time = std::chrono::steady_clock::now();

	unsigned int STATE_SIZE = 0;
	unsigned int NUM_QUBITS = 0;
	if (argc >= 2)
	{
		NUM_QUBITS = std::stoi(argv[1]);
		NUM_QUBITS = (NUM_QUBITS < 15) ? NUM_QUBITS : 10;
		STATE_SIZE = std::pow(2, NUM_QUBITS) ;
	}
	else
	{
		NUM_QUBITS = 10;
		STATE_SIZE = 1024;
	}

	cout << "NUMBER QUBITS = " << NUM_QUBITS << ", STATE SIZE = " << STATE_SIZE << endl;
	// STEP 1: BASIC LIBRARY CODE
	std::default_random_engine test_generator;

	//std::uniform_real_distribution<double> distribution(0.0, 1.0);
	std::uniform_int_distribution<int> distribution(0, 100);

	vector<vector<complex<double>>> IN_MAT_RAND(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));

	for (unsigned int i = 0; i < STATE_SIZE; i++)
	{
		for (unsigned int j = i; j < STATE_SIZE; j++)
		{
			IN_MAT_RAND[i][j] = distribution(test_generator);
			IN_MAT_RAND[j][i] = IN_MAT_RAND[i][j];
		}
	}
	
	Matrix A(IN_MAT_RAND);

	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	A.get_eigenvalues();

	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " msec" << endl;



	return 0;
}
