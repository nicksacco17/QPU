
#include <iostream>
#include <iomanip>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <functional>
#include <random>
#include <typeinfo>
#include "../include/Operator.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;
using std::string;

Operator::Operator() : Matrix()
{
	m_eigenvalues = { 1, 1 };
	m_eigenvectors = { {1, 0}, {0, 1} };
}

Operator::Operator(unsigned int in_row, unsigned int in_col) : Matrix(in_row, in_col)
{
	m_eigenvalues = vector<double>(m_num_row, 0.0);
	m_eigenvectors = vector<vector<complex<double>>>(m_num_row, vector<complex<double>>(m_num_col));
}

Operator::Operator(const vector<vector<complex<double>>>& in_mat) : Matrix(in_mat)
{
	m_eigenvalues = vector<double>(m_num_row, 0.0);
	m_eigenvectors = vector<vector<complex<double>>>(m_num_row, vector<complex<double>>(m_num_col));
}

Operator::Operator(const vector<complex<double>>& in_vec, unsigned int in_row, unsigned int in_col) : Matrix(in_vec, in_row, in_col)
{
	m_eigenvalues = vector<double>(m_num_row, 0.0);
	m_eigenvectors = vector<vector<complex<double>>>(m_num_row, vector<complex<double>>(m_num_col));
}

Operator::Operator(string mat_type, unsigned int in_dim, double lower_range, double upper_range, long unsigned int seed)
{
#ifdef USE_GPU
	cout << "I AM ON THE GPU" << endl;
	cout << "CUDA RANDOM GENERATION GOES HERE!" << endl;
#else
	m_dim = in_dim;
	m_num_row = m_dim;
	m_num_col = m_dim;

	m_eigenvalues = vector<double>(m_num_row, 0.0);
	m_eigenvectors = vector<vector<complex<double>>>(m_num_row, vector<complex<double>>(m_num_col));

	// Reset matrix
	m_mat = vector<complex<double>>(m_num_row * m_num_col, 0.0);

	double real_part = 0.0;
	double imag_part = 0.0;

	complex<double> rand_num = 0.0;

	std::default_random_engine RAND_NUM_GENERATOR{ static_cast<long unsigned int>(seed) };

	// Real Int
	if (mat_type == "R-I")
	{
		std::uniform_int_distribution<int> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					rand_num = complex<double>(real_part, 0.0);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = rand_num;
				}
			}
		}
	}
	// Real Float
	else if (mat_type == "R-F")
	{
		std::uniform_real_distribution<float> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					rand_num = complex<double>(real_part, 0.0);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = rand_num;
				}
			}
		}
	}
	// Real Double
	else if (mat_type == "R-D")
	{
		std::uniform_real_distribution<double> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					rand_num = complex<double>(real_part, 0.0);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = rand_num;
				}
			}
		}
	}

	// Complex Int
	else if (mat_type == "C-I")
	{
		std::uniform_int_distribution<int> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					imag_part = distribution(RAND_NUM_GENERATOR);
					rand_num = complex<double>(real_part, imag_part);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = std::conj(rand_num);
				}
			}
		}
	}
	// Complex Float
	else if (mat_type == "C-F")
	{
		std::uniform_real_distribution<float> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					imag_part = distribution(RAND_NUM_GENERATOR);
					rand_num = complex<double>(real_part, imag_part);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = std::conj(rand_num);
				}
			}
		}
	}
	// Complex Double
	else if (mat_type == "C-D")
	{
		std::uniform_real_distribution<double> distribution(lower_range, upper_range);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					imag_part = distribution(RAND_NUM_GENERATOR);
					rand_num = complex<double>(real_part, imag_part);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = std::conj(rand_num);
				}
			}
		}
	}
	else
	{
		std::uniform_int_distribution<int> distribution(0, 100);
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			for (unsigned int j = i; j < m_num_col; j++)
			{
				real_part = distribution(RAND_NUM_GENERATOR);

				// For Hermitian Operators, need to keep the diagonal real
				if (i == j)
				{
					rand_num = complex<double>(real_part, 0.0);
					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
				}
				// Else off diagnonal elements need to be Hermitian conjugates
				else
				{
					// Set imaginary part only if generating complex numbers
					rand_num = complex<double>(real_part, 0.0);

					m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = rand_num;
					m_mat.at(RC_TO_INDEX(j, i, m_num_col)) = rand_num;
				}
			}
		}
	}

#endif

}

Operator::~Operator()
{
	m_mat.clear();
	m_eigenvalues.clear();
	m_eigenvectors.clear();
}

Operator& Operator::operator=(const Matrix& mat)
{
	if (this != &mat)
	{
		this->m_num_row = mat.get_num_rows();
		this->m_num_col = mat.get_num_cols();
		this->m_mat = mat.get_row_order_mat();
	}

	return *this;
}

vector<double> Operator::get_eigenvalues()
{
	return m_eigenvalues;
}

double Operator::get_eigenvalue(unsigned int index)
{
	return m_eigenvalues.at(index);
}

vector<vector<complex<double>>> Operator::get_eigenvectors()
{
	return m_eigenvectors;
}

vector<complex<double>> Operator::get_eigenvector(unsigned int index)
{
	return m_eigenvectors.at(index);
}

Operator Operator::get_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2)
{
	Operator SUB_MAT;

	unsigned int NUM_ROWS = (row2 - row1) + 1;
	unsigned int NUM_COLS = (col2 - col1) + 1;

	vector<complex<double>> sub_mat_elements;

	for (unsigned int i = row1; i <= row2; i++)
	{
		for (unsigned int j = col1; j <= col2; j++)
		{
			sub_mat_elements.push_back(this->m_mat[RC_TO_INDEX(i, j, m_num_col)]);
		}
	}

	SUB_MAT.set_matrix(sub_mat_elements, NUM_ROWS, NUM_COLS);
	return SUB_MAT;
}

void Matrix::calc_eigenvalues()
{
#ifdef USE_GPU

	// CUDA STATUS, HANDLE, ERROR
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	int* d_info = NULL;
	int h_info = 0;

	const unsigned int NUM_ROW = m_num_row;
	const unsigned int NUM_COL = m_num_col;

	//double* EIG_VAL = new double [NUM_ROW];

	//complex<double>* EIG_VEC = new complex<double> [NUM_ROW * NUM_COL];

	cuDoubleComplex* d_MAT = NULL;
	cuDoubleComplex* d_WORK_MAT = NULL;

	double* d_EIG_VAL = NULL;
	int NUM_BYTES = 0;

	// STEP 3: CREATE HANDLE
	cusolver_status = cusolverDnCreate(&cusolverH);

	if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
	{
		cout << "[ERROR]: COULD NOT CREATE HANDLE" << endl;
		return;
	}

	// STEP 4: DATA ALLOCATION ON DEVICE
	// WILL COMPARE WITH CREATE VECTOR/MATRIX FROM CUBLAS

	cudaStat1 = cudaMalloc((void**)&d_MAT, sizeof(complex<double>) * (NUM_ROW * NUM_COL));
	cudaStat2 = cudaMalloc((void**)&d_EIG_VAL, sizeof(double) * NUM_ROW);
	cudaStat3 = cudaMalloc((void**)&d_info, sizeof(int));

	if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT ALLOCATE MEMORY ON DEVICE" << endl;
		return;
	}

	// STEP 5: TRANSFER DATA HOST --> DEVICE

	cudaStat1 = cudaMemcpy(d_MAT, this->get_col_order_mat().data(), sizeof(complex<double>) * (NUM_ROW * NUM_COL), cudaMemcpyHostToDevice);

	if (cudaStat1 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT TRANSFER DATA FROM HOST TO DEVICE" << endl;
		return;
	}

	// STEP 6: ALGORITHM CONFIGURATION

	// Mode configuration
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	// Calculate extra buffer space for host - THERE LOOKS TO BE A BETTER FUNCTION TO DO THIS THAT ALLOWS FOR MORE SPACE... DEFINITELY NEED TO CHECK IT OUT!

	cusolver_status = cusolverDnZheevd_bufferSize(cusolverH, jobz, uplo, NUM_ROW, d_MAT, NUM_COL, d_EIG_VAL, &NUM_BYTES);

	switch (cusolver_status)
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
		return;
	}

	// STEP 7: ALLOCATE ADDITIONAL WORK SPACE ON THE DEVICE

	cudaStat1 = cudaMalloc((void**)&d_WORK_MAT, sizeof(complex<double>) * NUM_BYTES);

	if (cudaStat1 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT ALLOCATE ADDITIONAL BUFFER SPACE ON DEVICE" << endl;
		return;
	}

	// STEP 8: COMPUTATION

	cusolver_status = cusolverDnZheevd(cusolverH, jobz, uplo, NUM_ROW, d_MAT, NUM_COL, d_EIG_VAL, d_WORK_MAT, NUM_BYTES, d_info);

	// Synchronize GPU work before returning control back to CPU
	cudaStat1 = cudaDeviceSynchronize();

	if (cusolver_status != CUSOLVER_STATUS_SUCCESS || cudaStat1 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT PERFORM CALCULATION" << endl;
		return;
	}

	// STEP 9: TRANSFER DATA DEVICE --> HOST

	cudaStat1 = cudaMemcpy(m_eigenvalues.data(), d_EIG_VAL, sizeof(double) * NUM_ROW, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(m_eigenvectors_UNFORMATTED.data(), d_MAT, sizeof(complex<double>) * (NUM_ROW * NUM_COL), cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

	if (cudaStat1 != cudaSuccess)
	{
		cout << "STAT 1 ERROR" << endl;
	}

	if (cudaStat2 != cudaSuccess)
	{
		cout << "STAT 2 ERROR" << endl;
	}

	if (cudaStat3 != cudaSuccess)
	{
		cout << "STAT 3 ERROR" << endl;
	}

	if (h_info != 0 || cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT TRANSFER DATA FROM DEVICE TO HOST" << endl;
		return;
	}

	// STEP 10: POST-PROCESSING

	// CHECK TO MAKE SURE THIS ORDERING MAKES SENSE, NOT SURE IF IT DOES - I THINK THIS SHOULD STORE THE EIGENVECTOR IN EACH ENTRY IN THE LIST
	for (unsigned int i = 0; i < m_num_col; i++)
	{
		for (unsigned int j = 0; j < m_num_row; j++)
		{
			m_eigenvectors.at(i).at(j) = m_eigenvectors_UNFORMATTED.at(RC_TO_INDEX(i, j, m_num_col));
		}
	}

	//printMatrix(NUM_ROW, NUM_COL, EIG_VEC, NUM_COL, "V");

	// STEP 11: MEMORY DEALLOCATION
	if (d_MAT)
	{
		cudaFree(d_MAT);
	}
	if (d_WORK_MAT)
	{
		cudaFree(d_WORK_MAT);
	}
	if (d_EIG_VAL)
	{
		cudaFree(d_EIG_VAL);
	}
	if (d_info)
	{
		cudaFree(d_info);
	}

	if (cusolverH)
	{
		cusolverDnDestroy(cusolverH);
	}

	cudaDeviceReset();

	return;
#else
	cout << "EIGVAL ON CPU..." << endl;
#endif
}

// e^A = P^-1 * e*lambda * P
void Operator::exponential()
{
	if (m_num_row == m_num_col)
	{

#ifdef USE_GPU

		this->calc_eigens();

		Operator MAT_DIAG(m_num_row, m_num_col);

		MAT_DIAG.createIdentityMatrix();

		for (unsigned int k = 0; k < m_num_row; k++)
		{
			MAT_DIAG.m_mat.at(RC_TO_INDEX(k, k, m_num_col)) = std::exp(m_eigenvalues.at(k));
		}

		Operator P(this->get_eigenvectors());
		P.transpose();

		Operator P_INV;
		P_INV = P;
		P_INV.inverse();

		Operator MAT_EXP;

		MAT_EXP = (P * (MAT_DIAG * P_INV));


		this->m_mat = MAT_EXP.m_mat;

#else
		cout << "MATRIX EXPONENTIAL ON CPU..." << endl;
#endif
	}
}

// Adapted from CUDA documentation and Stack Overflow
void Operator::inverse()
{
	if (m_num_row == m_num_col)
	{
#ifdef USE_GPU

		// CUDA STATUS, HANDLE, ERROR

		cublasStatus_t CUBLAS_STATUS = CUBLAS_STATUS_SUCCESS;
		cublasHandle_t CUBLAS_HANDLE = NULL;

		cudaError_t cudaStat1 = cudaSuccess;
		cudaError_t cudaStat2 = cudaSuccess;
		cudaError_t cudaStat3 = cudaSuccess;

		unsigned int DIM = m_num_row;
		unsigned int DIM_SQ = DIM * DIM;

		int* h_info = (int*)malloc(1 * sizeof(int));
		int* d_info;

		int* d_pivot;

		complex<double>* d_MAT = NULL;
		complex<double>* d_MAT_INV = NULL;
		cuDoubleComplex** d_pointers = NULL;
		cuDoubleComplex** d_pointers_res = NULL;

		// STEP 3: CREATE HANDLE
		//cusolver_status = cusolverDnCreate(&cusolverH);

		CUBLAS_STATUS = cublasCreate(&CUBLAS_HANDLE);

		if (CUBLAS_STATUS != CUBLAS_STATUS_SUCCESS)
		{
			cout << "[ERROR]: COULD NOT CREATE HANDLE" << endl;
			return;
		}

		// STEP 4: DATA ALLOCATION ON DEVICE
		// WILL COMPARE WITH CREATE VECTOR/MATRIX FROM CUBLAS

		cudaStat1 = cudaMalloc((void**)&d_MAT, sizeof(complex<double>) * (DIM * DIM));
		cudaStat2 = cudaMalloc((void**)&d_MAT_INV, sizeof(complex<double>) * (DIM * DIM));
		cudaStat3 = cudaMalloc((void**)&d_info, 1 * sizeof(int));
		cudaStat3 = cudaMalloc((void**)&d_pivot, sizeof(int) * DIM);
		cudaStat3 = cudaMalloc((void**)&d_pointers, 1 * sizeof(complex<double>*));
		cudaStat3 = cudaMalloc((void**)&d_pointers_res, 1 * sizeof(complex<double>*));

		if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess)
		{
			cout << "[ERROR]: COULD NOT ALLOCATE MEMORY ON DEVICE" << endl;
			return;
		}

		// STEP 5: TRANSFER DATA HOST --> DEVICE

		complex<double>** h_pointers = (complex<double>**)malloc(sizeof(complex<double>*));
		h_pointers[0] = d_MAT;

		complex<double>** h_pointers_res = (complex<double>**)malloc(sizeof(complex<double>*));
		h_pointers_res[0] = d_MAT_INV;

		cudaStat1 = cudaMemcpy(d_MAT, this->get_col_order_mat().data(), sizeof(complex<double>) * (DIM * DIM), cudaMemcpyHostToDevice);
		cudaStat2 = cudaMemcpy(d_pointers, h_pointers, 1 * sizeof(complex<double>*), cudaMemcpyHostToDevice);
		cudaStat3 = cudaMemcpy(d_pointers_res, h_pointers_res, 1 * sizeof(complex<double>*), cudaMemcpyHostToDevice);

		if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess)
		{
			cout << "[ERROR]: COULD NOT TRANSFER DATA FROM HOST TO DEVICE" << endl;
			return;
		}

		// STEP 6: ALGORITHM CONFIGURATION

		// Mode configuration
		//cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
		//cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

		// Calculate extra buffer space for host

		CUBLAS_STATUS = cublasZgetrfBatched(CUBLAS_HANDLE, DIM, d_pointers, DIM, d_pivot, d_info, 1);
		//CUBLAS_STATUS = cublasZgetrfBatched(CUBLAS_HANDLE, DIM, (cuDoubleComplex**) &d_MAT, DIM, d_pivot, d_info, 1);

		cudaStat1 = cudaMemcpy(h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

		cout << "INFO: " << h_info[0] << endl;

		//cusolver_status = cusolverDnZheevd_bufferSize(cusolverH, jobz, uplo, NUM_ROW, d_A, NUM_COL, d_EIG_VAL, &NUM_BYTES);

		switch (CUBLAS_STATUS)
		{
		case CUBLAS_STATUS_NOT_INITIALIZED:
			cout << "NOT INIT" << endl;
			break;
		case CUBLAS_STATUS_INVALID_VALUE:
			cout << "INVALID VALUE" << endl;
			break;
		case CUBLAS_STATUS_ARCH_MISMATCH:
			cout << "ARCH" << endl;
			break;
		case CUBLAS_STATUS_EXECUTION_FAILED:
			cout << "COULD NOT LAUNCH ON GPU" << endl;
			break;
		case CUBLAS_STATUS_SUCCESS:
			cout << "INVERSE CALCULATED" << endl;
			break;
		default: cout << "?????" << endl;
		}

		if (CUBLAS_STATUS != CUBLAS_STATUS_SUCCESS)
		{
			cout << "[ERROR]: BATCH ALGO 1 INCORRECT" << endl;
			return;
		}

		CUBLAS_STATUS = cublasZgetriBatched(CUBLAS_HANDLE, DIM, d_pointers, DIM, d_pivot, d_pointers_res, DIM, d_info, 1);

		cudaStat1 = cudaMemcpy(&h_info, &d_info, sizeof(int), cudaMemcpyDeviceToHost);

		cout << "INFO: " << h_info[0] << endl;

		if (CUBLAS_STATUS != CUBLAS_STATUS_SUCCESS)
		{
			cout << "[ERROR]: BATCH ALGO 2 INCORRECT" << endl;
			return;
		}

		cudaStat1 = cudaDeviceSynchronize();

		if (cudaStat1 != cudaSuccess)
		{
			cout << "[ERROR]: COULD NOT SYNCHRONIZE" << endl;
			return;
		}

		// STEP 9: TRANSFER DATA DEVICE --> HOST

		cudaStat2 = cudaMemcpy(this->m_mat.data(), d_MAT_INV, sizeof(complex<double>) * (DIM * DIM), cudaMemcpyDeviceToHost);
		//cudaStat3 = cudaMemcpy(&h_info, &d_info, sizeof(int), cudaMemcpyDeviceToHost);

		//if (cudaStat1 != cudaSuccess)
		//{
		//	cout << "STAT 1 ERROR" << endl;
		//}

		if (cudaStat2 != cudaSuccess)
		{
			cout << "STAT 2 ERROR" << endl;
		}

		//if (cudaStat3 != cudaSuccess)
		//{
		//	cout << "STAT 3 ERROR" << endl;
		//}

		if (h_info[0] != 0 || cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess)// || cudaStat3 != cudaSuccess)
		{
			cout << "[ERROR]: COULD NOT TRANSFER DATA FROM DEVICE TO HOST" << endl;
			return;
		}

		// STEP 10: POST-PROCESSING

		//printMatrix(NUM_ROW, NUM_COL, EIG_VEC, NUM_COL, "V");

		this->transpose();

		// STEP 11: MEMORY DEALLOCATION
		if (d_MAT)
		{
			cudaFree(d_MAT);
		}
		if (d_MAT_INV)
		{
			cudaFree(d_MAT_INV);
		}
		if (d_info)
		{
			//cudaFree(d_info);
		}
		if (CUBLAS_HANDLE)
		{
			cublasDestroy(CUBLAS_HANDLE);
		}

		cudaDeviceReset();

		return;
#else
	cout << "INVERSE ON CPU..." << endl;
#endif
	}
}

void Operator::print() const
{
	cout << "---------- PRINT OPERATOR ----------" << endl;

	cout << "DIMENSION: (" << m_num_row << " x " << m_num_col << ")" << endl;
	cout << "TRACE: " << m_trace << endl;
	cout << "DETERMINANT: " << m_determinant << endl;

	cout << "ELEMENTS:\n" << endl;

	for (unsigned int i = 0; i < m_num_row; i++)
	{
		cout << "| ";
		for (unsigned int j = 0; j < m_num_col; j++)
		{
			//cout << "(i, j) = (" << i << ", " << j << ") = " << RC_TO_INDEX(i, j, m_num_col) << endl;
			cout << std::fixed << std::setprecision(6) << m_mat.at(RC_TO_INDEX(i, j, m_num_col)) << " ";
		}
		cout << "|" << endl;
	}

	cout << "\n---------- PRINT OPERATOR ----------" << endl;
}