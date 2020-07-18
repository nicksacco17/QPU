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
#include "./include/Matrix.h"
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

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++)
    {
        for(int col = 0 ; col < n ; col++)
        {
            printf("%s(%d,%d) = %f\n", name, row, col, A[row + col * lda]);
        }
    }
}

int main(int argc, char*argv[])
{
	std::chrono::steady_clock::time_point total_start_time = std::chrono::steady_clock::now();

	unsigned int STATE_SIZE = 0;
	if (argc >= 2)
	{
		STATE_SIZE = std::stoi(argv[1]);
		cout << "STATE SIZE = " << STATE_SIZE << endl;
	}
	else
	{
		STATE_SIZE = 32;
		cout << "STATE SIZE = " << STATE_SIZE << endl;
	}

	// STEP 1: BASIC LIBRARY CODE
	std::default_random_engine test_generator;

	//std::uniform_real_distribution<double> distribution(0.0, 1.0);
	std::uniform_int_distribution<int> distribution(0, 100);

	vector<vector<complex<double>>> IN_MAT_RAND(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));

	std::chrono::steady_clock::time_point rand_gen_start_time = std::chrono::steady_clock::now();

	for (unsigned int i = 0; i < STATE_SIZE; i++)
	{
		for (unsigned int j = i; j < STATE_SIZE; j++)
		{
			IN_MAT_RAND[i][j] = distribution(test_generator);
			IN_MAT_RAND[j][i] = IN_MAT_RAND[i][j];
		}
	}
	
	std::chrono::steady_clock::time_point rand_gen_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL RAND GEN TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(rand_gen_stop_time - rand_gen_start_time).count() << " msec" << endl;

	// CUDA STATUS, HANDLE, ERROR

	cusolverDnHandle_t cusolverH = NULL;
    	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    	
	cudaError_t cudaStat1 = cudaSuccess;
    	cudaError_t cudaStat2 = cudaSuccess;
    	cudaError_t cudaStat3 = cudaSuccess;
    	int* devInfo = NULL;
	int info_gpu = 0;

	const unsigned int NUM_ROW = STATE_SIZE;
	const unsigned int NUM_COL = STATE_SIZE;

	std::chrono::steady_clock::time_point var_create_start_time = std::chrono::steady_clock::now();

	double* EIG_VAL = new double [NUM_ROW];

	complex<double>* EIG_VEC = new complex<double> [NUM_ROW * NUM_COL];

	//complex<double> EIG_VEC[NUM_ROW * NUM_COL];

	cuDoubleComplex* d_A = NULL;
	double* d_EIG_VAL = NULL;
	cuDoubleComplex* d_work_A = NULL;
	long int NUM_BYTES = 0;

	Matrix* B = new Matrix(IN_MAT_RAND);

	//Matrix B(IN_MAT_RAND);
	//B.print();

	vector<complex<double>> COL_MAT = B->get_col_order_mat();

	std::chrono::steady_clock::time_point var_create_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL VAR CREATE TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(var_create_stop_time - var_create_start_time).count() << " msec" << endl;

	//Matrix C(COL_MAT, STATE_SIZE, STATE_SIZE);
	//C.print();

	// STEP 3: CREATE HANDLE
	cusolver_status = cusolverDnCreate(&cusolverH);

	if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
	{
		cout << "[ERROR]: COULD NOT CREATE HANDLE" << endl;
		return 0;
	}

	// STEP 4: DATA ALLOCATION ON DEVICE
	// WILL COMPARE WITH CREATE VECTOR/MATRIX FROM CUBLAS

	std::chrono::steady_clock::time_point device_malloc_start_time = std::chrono::steady_clock::now();

	cudaStat1 = cudaMalloc((void**) &d_A, sizeof(complex<double>) * (B->get_num_rows() * B->get_num_cols()));
	cudaStat2 = cudaMalloc((void**) &d_EIG_VAL, sizeof(double) * B->get_num_rows());
	cudaStat3 = cudaMalloc((void**) &devInfo, sizeof(int));

	std::chrono::steady_clock::time_point device_malloc_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL MALLOC TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(device_malloc_stop_time - device_malloc_start_time).count() << " msec" << endl;

	if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT ALLOCATE MEMORY ON DEVICE" << endl;
		return 0;
	}

	// STEP 5: TRANSFER DATA HOST --> DEVICE
	
	std::chrono::steady_clock::time_point host_device_transfer_start_time = std::chrono::steady_clock::now();
	
	cudaStat1 = cudaMemcpy(d_A, B->get_col_order_mat().data(), sizeof(complex<double>) * (B->get_num_rows() * B->get_num_cols()), cudaMemcpyHostToDevice);

	std::chrono::steady_clock::time_point host_device_transfer_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL HOST --> DEVICE TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(host_device_transfer_stop_time - host_device_transfer_start_time).count() << " msec" << endl;

	if (cudaStat1 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT TRANSFER DATA FROM HOST TO DEVICE" << endl;
		return 0;
	}

	// STEP 6: ALGORITHM CONFIGURATION

	// Mode configuration
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	// Calculate extra buffer space for host
	
	std::chrono::steady_clock::time_point buffer_start_time = std::chrono::steady_clock::now();

	cusolver_status = cusolverDnZheevd_bufferSize(cusolverH, jobz, uplo, NUM_ROW, d_A, NUM_COL, d_EIG_VAL, &(int*)NUM_BYTES);

	switch(cusolver_status)
	{
		case CUSOLVER_STATUS_NOT_INITIALIZED: 
			cout << "NOT INIT" << endl; 
			break;
		case CUSOLVER_STATUS_INVALID_VALUE:
			cout << "INVALID VALUE" << endl;
			break;
		case CUSOLVER_STATUS_ARCH_MISMATCH:
			cout << "ARCH" << endl;
			break;
		case CUSOLVER_STATUS_INTERNAL_ERROR:
			cout << "INTERNAL ERROR" << endl;
			break;
		case CUSOLVER_STATUS_SUCCESS:
			cout << "BUFFER ALLOCATED" << endl;
			break;
		default: cout << "?????" << endl;		
	}

	cout << "NUM BYTES: " << NUM_BYTES << endl;

	if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
	{
		cout << "[ERROR]: COULD NOT DETERMINE EXTRA BUFFER SPACE ON DEVICE" << endl;
		return 0;
	}

	// STEP 7: ALLOCATE ADDITIONAL WORK SPACE ON THE DEVICE

	cudaStat1 = cudaMalloc((void**) &d_work_A, sizeof(complex<double>) * NUM_BYTES); 
	
	std::chrono::steady_clock::time_point buffer_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL BUFFER TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(buffer_stop_time - buffer_start_time).count() << " msec" << endl;

	if (cudaStat1 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT ALLOCATE ADDITIONAL BUFFER SPACE ON DEVICE" << endl;
		return 0;
	}

	// STEP 8: COMPUTATION

	std::chrono::steady_clock::time_point comp_start_time = std::chrono::steady_clock::now();
	
	cusolver_status = cusolverDnZheevd(cusolverH, jobz, uplo, NUM_ROW, d_A, NUM_COL, d_EIG_VAL, d_work_A, NUM_BYTES, devInfo);
	
	std::chrono::steady_clock::time_point comp_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL COMPUTATION TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(comp_stop_time - comp_start_time).count() << " msec" << endl;

	// Synchronize GPU work before returning control back to CPU
	std::chrono::steady_clock::time_point sync_start_time = std::chrono::steady_clock::now();

	cudaStat1 = cudaDeviceSynchronize();

	std::chrono::steady_clock::time_point sync_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL SYNC TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(sync_stop_time - sync_start_time).count() << " msec" << endl;

	if (cusolver_status != CUSOLVER_STATUS_SUCCESS || cudaStat1 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT PERFORM CALCULATION" << endl;
		return 0;
	}

	// STEP 9: TRANSFER DATA DEVICE --> HOST

	std::chrono::steady_clock::time_point device_host_transfer_start_time = std::chrono::steady_clock::now();

	cudaStat1 = cudaMemcpy(EIG_VAL, d_EIG_VAL, sizeof(double) * B->get_num_rows(), cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(EIG_VEC, d_A, sizeof(complex<double>) * (B->get_num_rows() * B->get_num_cols()), cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

	std::chrono::steady_clock::time_point device_host_transfer_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL DEVICE --> HOST TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(device_host_transfer_stop_time - device_host_transfer_start_time).count() << " msec" << endl;

	if (info_gpu != 0 || cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT TRANSFER DATA FROM DEVICE TO HOST" << endl;
		return 0;
	}

	// STEP 10: POST-PROCESSING

	cout << "EIGENVALUES: " << endl;

	/*for (int i = 0; i < B.get_num_rows(); i++)
	{
		cout << "lambda[" << i << "] = " << EIG_VAL[i] << endl;
	}*/

	//cout << "EIGENVECTORS: " << endl;

	//printMatrix(NUM_ROW, NUM_COL, EIG_VEC, NUM_COL, "V");

	// STEP 11: MEMORY DEALLOCATION

	if (d_A)
		cudaFree(d_A);
	if (d_EIG_VAL)
		cudaFree(d_EIG_VAL);
	if (devInfo)
		cudaFree(devInfo);
	if (d_work_A)
		cudaFree(d_work_A);
	if (cusolverH)
		cusolverDnDestroy(cusolverH);
	cudaDeviceReset();

	delete B;
	delete EIG_VAL;
	delete EIG_VEC;

    	/*    
	// Calculate extra buffer space on host
	cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork);

	// Step 4: compute spectrum
	// Step 4: Computation
    	cusolver_status = cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, de
	*/

	std::chrono::steady_clock::time_point total_stop_time = std::chrono::steady_clock::now();
	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(total_stop_time - total_start_time).count() << " msec" << endl;

    	return 0;
}


	//vector<vector<complex<double>>> IN_MAT = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

	//Matrix E(IN_MAT);

	//E.print();

	/*unsigned int DIM_ROW = 3;
	unsigned int DIM_COL = 3;

	Matrix E(DIM_ROW, DIM_COL);

	for (unsigned int i = 0; i < DIM_ROW; i++)
	{
		for (unsigned int j = 0; j < DIM_COL; j++)
		{
			E.set_element(i, j, i);
		}
	}

	E.print();*/

