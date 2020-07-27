
#include "../include/Matrix.h"
#include "../include/Utility.h"

#include <iostream>
#include <algorithm>
#include <functional>
#include <chrono>

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#endif

using std::cout;
using std::endl;

/* ****************************** CONSTRUCTORS ****************************** */

Matrix::Matrix()
{
	m_num_row = 2;
	m_num_col = 2;
	m_dim = 2;

	m_mat = { 1, 0, 0, 1 };

	m_determinant = 1;
	m_trace = 2;

	m_eigenvalues = {1, 1};
	m_eigenvectors = {{1, 0}, {0, 1}};

	//m_cache.update_trace_flag = false;
	//m_cache.update_determinant_flag = false;
	//m_cache.update_eig_flag = false;
}

Matrix::Matrix(unsigned int in_row, unsigned int in_col)
{
	m_num_row = in_row;
	m_num_col = in_col;

	m_dim = (m_num_row == m_num_col) ? m_num_row : -1;
	
	m_mat = vector<complex<double>>(m_num_row * m_num_col, 0.0);

	m_determinant = 9999;
	m_trace = 9999;
}

Matrix::Matrix(const vector<vector<complex<double>>>& in_mat)
{
	m_num_row = in_mat.size();
	m_num_col = in_mat.at(0).size();

	m_dim = (m_num_row == m_num_col) ? m_num_row : -1;

	for (unsigned int i = 0; i < in_mat.size(); i++)
	{
		m_mat.insert(m_mat.end(), in_mat.at(i).begin(), in_mat.at(i).end());
	}

	m_determinant = 9999;
	m_trace = 9999;

	//m_cache.update_trace_flag = true;
	//m_cache.update_determinant_flag = true;
	//m_cache.update_eig_flag = true;
}

Matrix::Matrix(const vector<complex<double>>& in_vec, unsigned int in_row, unsigned int in_col)
{
	m_mat = in_vec;

	m_num_row = in_row;
	m_num_col = in_col;

	//m_cache.update_trace_flag = true;
	//m_cache.update_determinant_flag = true;
	//m_cache.update_eig_flag = true;
}

Matrix::~Matrix()
{
	m_mat.clear();
	m_eigenvalues.clear();
	m_eigenvectors.clear();
}

/* ************************************************************************** */

/* ******************************* OPERATORS ******************************** */

// Assignment (A = B)
Matrix& Matrix::operator=(const Matrix& mat)					
{
	if (this != &mat)
	{
		this->m_num_row = mat.m_num_row;
		this->m_num_col = mat.m_num_col;
		this->m_mat = mat.m_mat;
	}

	return *this;
}

// (Strict) Equality (A == B)
bool Matrix::operator==(const Matrix& mat)					
{
	if (this->m_num_row == mat.m_num_row && this->m_num_col == mat.m_num_col)
	{
		for (unsigned int i = 0; i < this->m_mat.size(); i++)
		{
			if (this->m_mat.at(i) != mat.m_mat.at(i))
			{
				return false;
			}
		}
	}
	else
	{
		return false;
	}

	return true;
}

// Not Equal (A != B)
bool Matrix::operator!=(const Matrix& mat)						
{
	return !(*this == mat);
}

// Scalar Addition (A += B)
Matrix& Matrix::operator+=(const Matrix& mat)					
{

#ifdef USE_GPU
	cout << "GPU MATRIX ADDITION..." << endl;
#else
	std::transform(this->m_mat.begin(), this->m_mat.end(), mat.m_mat.begin(), this->m_mat.begin(), std::plus<complex<double>>());
#endif

	return *this;
}

// Scalar Addition (C = A + B)
const Matrix Matrix::operator+(const Matrix& mat) const		
{
	Matrix TEMP = *this;
	TEMP += mat;

	return TEMP;
}

// Scalar Subtraction (A -= B)
Matrix& Matrix::operator-=(const Matrix& mat)					
{

#ifdef USE_GPU
	cout << "GPU MATRIX SUBTRACTION..." << endl;
#else
	std::transform(this->m_mat.begin(), this->m_mat.end(), mat.m_mat.begin(), this->m_mat.begin(), std::minus<complex<double>>());
#endif

	return *this;
}

// Scalar Subtraction (C = A - B)
const Matrix Matrix::operator-(const Matrix& mat) const		
{
	Matrix TEMP = *this;
	TEMP -= mat;

	return TEMP;
}

// Scalar Multiplication (A *= a)
Matrix& Matrix::operator*=(const complex<double> alpha)			
{

#ifdef USE_GPU

	cout << "GPU SCALAR MULTIPLICATION..." << endl;
	cudaError_t CUDA_STATUS;
	cublasStatus_t CUBLAS_STATUS;
	cublasHandle_t CUDA_HANDLE;

	cuDoubleComplex* d_MAT = NULL;

	CUDA_STATUS = cudaMalloc((void**) &d_MAT, sizeof(complex<double>) * m_mat.size());
	CUDA_HANDLE = cublasCreate(&CUDA_HANDLE);

	CUDA_STATUS = cudaMemcpy(d_MAT, COL_MAT.data(), sizeof(complex<double>) * (m_num_row * m_num_col), cudaMemcpyHostToDevice);

	CUBLAS_STATUS = cublasZscal(CUDA_HANDLE, m_mat.size(), &alpha, m_mat.data(), 1);

#else
	std::transform(this->m_mat.begin(), this->m_mat.end(), this->m_mat.begin(), std::bind1st(std::multiplies<complex<double>>(), alpha));
#endif

	return *this;
}

// Scalar Multiplication (B = A * a)
const Matrix Matrix::operator*(const complex<double> alpha) const	
{
	Matrix TEMP = *this;
	TEMP *= alpha;

	return TEMP;
}

// Scalar Division (A *= 1/a)
Matrix& Matrix::operator/=(const complex<double> alpha)
{

#ifdef USE_GPU
	cout << "GPU SCALAR DIVISION..." << endl;
#else
	if (std::real(alpha) > 1e-10 && std::imag(alpha) > 1e-10)
	{
		std::transform(this->m_mat.begin(), this->m_mat.end(), this->m_mat.begin(), std::bind1st(std::multiplies<complex<double>>(), (1.0 / alpha)));
	}
#endif

	return *this;
}

// Scalar Division (B = A * 1/a)
const Matrix Matrix::operator/(const complex<double> a) const
{
	Matrix TEMP = *this;
	TEMP /= a;

	return TEMP;
}

// Matrix Multiplication (A *= B)
Matrix& Matrix::operator*=(const Matrix& mat)					
{

#ifdef USE_GPU
	cout << "MATRIX MULTIPLICATION THROUGH GPU" << endl;
#else
	if (this != &mat && this->m_num_col == mat.m_num_row)
	{
		unsigned int NUM_ELEMENTS = this->m_num_row * mat.m_num_col;
		unsigned int COMMON_DIM = this->m_num_col;
		vector<complex<double>> NEW_MAT(NUM_ELEMENTS, 0.0);

		for (unsigned int i = 0; i < this->m_num_row; i++)
		{
			//cout << "ROW: " << i << endl;
			for (unsigned int k = 0; k < COMMON_DIM; k++)
			{
				for (unsigned int j = 0; j < mat.m_num_col; j++)
				{
					NEW_MAT.at(RC_TO_INDEX(i, j, m_num_col)) += (this->m_mat.at(RC_TO_INDEX(i, k, m_num_col)) * mat.m_mat.at(RC_TO_INDEX(k, j, m_num_col)));
				}
			}
		}

		this->m_mat = NEW_MAT;
		this->m_num_row = this->m_num_row;
		this->m_num_col = mat.m_num_col;
	}
#endif

	return *this;
}

// Matrix Multiplication (C = A * B)
const Matrix Matrix::operator*(const Matrix& mat) const		
{
	Matrix TEMP = *this;
	TEMP *= mat;

	return TEMP;
}

/* ************************************************************************** */

/* ************************** ACCESSORS & MUTATORS ************************** */

complex<double> Matrix::get_element(unsigned int row, unsigned int col) const
{
	return m_mat.at(RC_TO_INDEX(row, col, m_num_col));
}

void Matrix::set_element(unsigned int row, unsigned int col, complex<double> in_value)
{
	m_mat.at(RC_TO_INDEX(row, col, m_num_col)) = in_value;
}

unsigned int Matrix::get_num_rows() const
{
	return m_num_row;
}

unsigned int Matrix::get_num_cols() const
{
	return m_num_col;
}

void Matrix::set_dims(unsigned int in_row, unsigned int in_col)
{
	m_num_row = in_row;
	m_num_col = in_col;

	m_dim = (m_num_row == m_num_col) ? m_num_row : -1;

	m_mat = vector<complex<double>>(m_num_row * m_num_col, 0.0);
}

vector<complex<double>> Matrix::get_row_order_mat() const
{
	return m_mat;
}

vector<complex<double>> Matrix::get_col_order_mat() const
{
	vector<complex<double>> COL_MAT;

	// For each column
	for (unsigned int j = 0; j < m_num_col; j++)
	{
		// Get all the elements in the row
		for (unsigned int i = 0; i < m_num_row; i++)
		{
			COL_MAT.push_back(m_mat.at(RC_TO_INDEX(i, j, m_num_col)));
		}
	}

	return COL_MAT;
}

vector<vector<complex<double>>> Matrix::get_matrix()
{
	return { {1, 0}, {1, 2} };
}

void Matrix::set_matrix(vector<vector<complex<double>>>& in_mat)
{
	m_num_row = in_mat.size();
	m_num_col = in_mat.at(0).size();

	m_dim = (m_num_row == m_num_col) ? m_num_row : -1;

	for (unsigned int i = 0; i < in_mat.size(); i++)
	{
		m_mat.insert(m_mat.end(), in_mat.at(i).begin(), in_mat.at(i).end());
	}

	m_determinant = 9999;
	m_trace = 9999;
}

void Matrix::set_matrix(vector<complex<double>>& in_vec, unsigned int in_row, unsigned int in_col)
{
	m_mat = in_vec;
	m_num_row = in_row;
	m_num_col = in_col;
}

Matrix Matrix::get_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2)
{
	Matrix SUB_MAT;

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

void Matrix::set_submatrix(unsigned int row1, unsigned int row2, unsigned int col1, unsigned int col2, const Matrix& submatrix)
{
	unsigned int COL_STRIDE = submatrix.m_num_col;

	//cout << RC_TO_INDEX()

	for (unsigned int i = row1; i <= row2; i++)
	{
		for (unsigned int j = col1; j <= col2; j++)
		{
			//cout << "(i, j) = (" << i - row1 << ", " << j - col1 << ") = " << submatrix.m_mat.at(RC_TO_INDEX(i - row1, j - col1)) << endl;
			this->m_mat.at(RC_TO_INDEX(i, j, m_num_col)) = submatrix.m_mat.at(RC_TO_INDEX(i - row1, j - col1, submatrix.m_num_col));
		}
	}
}

/* ************************************************************************** */

/* ******************************* FUNCTIONS ******************************** */

void Matrix::createIdentityMatrix()
{
	m_mat = vector<complex<double>>(m_num_row * m_num_col, 0.0);

	if (m_num_row == m_num_col)
	{
		for (unsigned int k = 0; k < m_num_row; k++)
		{
			m_mat.at(RC_TO_INDEX(k, k, m_num_col)) = 1;
		}
	}
}

void Matrix::transpose()
{
	unsigned int new_num_row = this->m_num_col;
	unsigned int new_num_col = this->m_num_row;

	complex<double> TEMP;

	// If square matrix
	if (m_num_row == m_num_col)
	{
		for (unsigned int row = 0; row < m_num_row - 1; row++)
		{
			for (unsigned int col = row + 1; col < m_num_col; col++)
			{
				TEMP = m_mat.at(RC_TO_INDEX(row, col, m_num_col));
				m_mat.at(RC_TO_INDEX(row, col, m_num_col)) = m_mat.at(RC_TO_INDEX(col, row, m_num_col));
				m_mat.at(RC_TO_INDEX(col, row, m_num_col)) = TEMP;
			}
		}
	}

	// Else if not square
	else
	{
		vector<complex<double>> NEW_MAT(new_num_row * new_num_col, 0.0);

		for (unsigned int row = 0; row < m_num_row; row++)
		{
			for (unsigned int col = 0; col < m_num_col; col++)
			{
				NEW_MAT.at(RC_TO_INDEX(col, row, new_num_col)) = m_mat.at(RC_TO_INDEX(row, col, m_num_col));
			}
		}
		m_mat = NEW_MAT;
	}

	m_num_row = new_num_row;
	m_num_col = new_num_col;
}

void Matrix::conjugate()
{

}

void Matrix::hermitian_conjugate()
{

}

void Matrix::inverse()
{

}

vector<complex<double>> Matrix::get_eigenvalues()
{
	return m_eigenvalues;
}

complex<double> Matrix::get_eigenvalue(unsigned int index)
{
	return m_eigenvalues.at(index);
}

vector<vector<complex<double>>> Matrix::get_eigenvectors()
{
	return m_eigenvectors;
}

vector<complex<double>> Matrix::get_eigenvector(unsigned int index)
{
	return m_eigenvectors.at(index);
}

complex<double> Matrix::get_determinant() const
{
	return m_determinant;
}

complex<double> Matrix::get_trace() const
{
	return m_trace;
}

/* ************************************************************************** */

/* ******************************** UTILITY ********************************* */

void Matrix::print() const
{
	cout << "---------- PRINT MATRIX ----------" << endl;

	cout << "DIMENSION: (" << m_num_row << " x " << m_num_col << ")" << endl;
	cout << "TRACE: " << m_trace << endl;
	cout << "DETERMINANT: " << m_determinant << endl;

	cout << "ELEMENTS:\n" << endl;

	for (unsigned int i = 0; i < m_num_row; i++)
	{
		cout << "| ";
		for (unsigned int j = 0; j < m_num_col; j++)
		{
			cout << "(i, j) = (" << i << ", " << j << ") = " << RC_TO_INDEX(i, j, m_num_col) << endl;
			cout << m_mat.at(RC_TO_INDEX(i, j, m_num_col)) << " ";
		}
		cout << "|" << endl;
	}

	cout << "\n---------- PRINT MATRIX ----------" << endl;
}

void Matrix::print_shape() const
{
	for (unsigned int i = 0; i < m_num_row; i++)
	{
		cout << "| ";
		for (unsigned int j = 0; j < m_num_col; j++)
		{
			if (iszero_print(m_mat.at(RC_TO_INDEX(i, j, m_num_col))))
			{
				cout << "0 ";
			}

			else if (i == j)
			{
				cout << "+ ";
			}

			else
			{
				cout << "* ";
			}
		}
		cout << "|" << endl;
	}
}

void Matrix::clear()
{
	m_mat = vector<complex<double>>(m_num_row * m_num_col, 0.0);
}

/* ************************************************************************** */


// CUDA EIGENVALUES
#ifdef USEGPU
void Matrix::calc_eigenvalues()
{
	// CUDA STATUS, HANDLE, ERROR
	cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    	
	cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    int* devInfo = NULL;
	int info_gpu = 0;

	const unsigned int NUM_ROW = m_num_row;
	const unsigned int NUM_COL = m_num_col;

	//double* EIG_VAL = new double [NUM_ROW];

	complex<double>* EIG_VEC = new complex<double> [NUM_ROW * NUM_COL];

	cuDoubleComplex* d_A = NULL;
	double* d_EIG_VAL = NULL;
	cuDoubleComplex* d_work_A = NULL;
	int NUM_BYTES = 0;

	vector<complex<double>> COL_MAT = this->get_col_order_mat();

	// STEP 3: CREATE HANDLE
	cusolver_status = cusolverDnCreate(&cusolverH);

	if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
	{
		cout << "[ERROR]: COULD NOT CREATE HANDLE" << endl;
		return;
	}

	// STEP 4: DATA ALLOCATION ON DEVICE
	// WILL COMPARE WITH CREATE VECTOR/MATRIX FROM CUBLAS

	std::chrono::steady_clock::time_point device_malloc_start_time = std::chrono::steady_clock::now();

	cudaStat1 = cudaMalloc((void**) &d_A, sizeof(complex<double>) * (m_num_row * m_num_col));
	cudaStat2 = cudaMalloc((void**) &d_EIG_VAL, sizeof(double) * m_num_row);
	cudaStat3 = cudaMalloc((void**) &devInfo, sizeof(int));

	if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT ALLOCATE MEMORY ON DEVICE" << endl;
		return;
	}

	// STEP 5: TRANSFER DATA HOST --> DEVICE

	cudaStat1 = cudaMemcpy(d_A, COL_MAT.data(), sizeof(complex<double>) * (m_num_row * m_num_col), cudaMemcpyHostToDevice);

	if (cudaStat1 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT TRANSFER DATA FROM HOST TO DEVICE" << endl;
		return;
	}

	// STEP 6: ALGORITHM CONFIGURATION

	// Mode configuration
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	// Calculate extra buffer space for host
	
	cusolver_status = cusolverDnZheevd_bufferSize(cusolverH, jobz, uplo, NUM_ROW, d_A, NUM_COL, d_EIG_VAL, &NUM_BYTES);

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
		return;
	}

	// STEP 7: ALLOCATE ADDITIONAL WORK SPACE ON THE DEVICE

	cudaStat1 = cudaMalloc((void**) &d_work_A, sizeof(complex<double>) * NUM_BYTES); 
	
	if (cudaStat1 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT ALLOCATE ADDITIONAL BUFFER SPACE ON DEVICE" << endl;
		return;
	}

	// STEP 8: COMPUTATION

	cusolver_status = cusolverDnZheevd(cusolverH, jobz, uplo, NUM_ROW, d_A, NUM_COL, d_EIG_VAL, d_work_A, NUM_BYTES, devInfo);
	
	// Synchronize GPU work before returning control back to CPU
	cudaStat1 = cudaDeviceSynchronize();

	if (cusolver_status != CUSOLVER_STATUS_SUCCESS || cudaStat1 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT PERFORM CALCULATION" << endl;
		return;
	}

	// STEP 9: TRANSFER DATA DEVICE --> HOST

	cudaStat1 = cudaMemcpy(m_eigenvalues.data(), d_EIG_VAL, sizeof(double) * m_num_row, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(EIG_VEC, d_A, sizeof(complex<double>) * (m_num_row * m_num_col), cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

	if (info_gpu != 0 || cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess)
	{
		cout << "[ERROR]: COULD NOT TRANSFER DATA FROM DEVICE TO HOST" << endl;
		return;
	}

	// STEP 10: POST-PROCESSING

	//printMatrix(NUM_ROW, NUM_COL, EIG_VEC, NUM_COL, "V");

	// STEP 11: MEMORY DEALLOCATION
	if (d_A)
	{
		cudaFree(d_A);
	}	
	if (d_EIG_VAL)
	{
		cudaFree(d_EIG_VAL);
	}
	if (devInfo)
	{
		cudaFree(devInfo);
	}
	if (d_work_A)
	{
		cudaFree(d_work_A);
	}
	if (cusolverH)
	{
		cusolverDnDestroy(cusolverH);
	}

	if (EIG_VEC)
	{
		delete EIG_VEC;
	}

	cudaDeviceReset();

	return;
}
#endif
