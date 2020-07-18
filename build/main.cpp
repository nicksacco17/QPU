
#include <iostream>
#include <string>
#include <ctime>
#include <random>

#include "../include/TestDrivers.h"
#include "../include/Operator.h"
#include "../include/State.h"
#include "../include/Pauli_Matrix.h"
#include "../include/LinearAlgebra.h"
#include "../include/Gate.h"
#include "../include/QuantumCircuit.h"

#include <vector>
#include <complex>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <fstream>

using std::cout;
using std::endl;
using std::cin;
using std::vector;
using std::complex;
using std::string;

using namespace std::complex_literals;

#define MAX_QUBITS 12
#define n 9
#define N 3
#define NUM_QUBITS 10
#define STATE_SIZE unsigned int(std::pow(2, NUM_QUBITS))

#define MAT_SIZE 10

void test()
{
	/*
	std::default_random_engine test_generator;

	std::uniform_int_distribution<int> distribution(0, 100);
	std::uniform_int_distribution<int> boolean_dist(0, 2);


	//vector<vector<complex<double>>> mat_2d_sq(MAT_SIZE, vector<complex<double>>(MAT_SIZE, 0.0));
	vector<vector<complex<double>>> mat_2d_sq = { {6, 5, 0}, {5, 1, 4}, {0, 4, 3} };
	vector<vector<complex<double>>> mat_2d_complex = { {6i, 5i, 0}, {5i, 1i, 4i}, {0, 4i, 3i} };
	vector<vector<complex<double>>> mat_2d_test = { {2, 3, 1, 0.5, 4}, {4, 5, 7, 0.1, 1}, {5, 3, 6, 19.2, 9}, {1, 4, 1, 4, 7}, {3, 1, 6, 2, 6} };

	vector<vector<complex<double>>> mat1_2d_rand(MAT_SIZE, vector<complex<double>>(MAT_SIZE, 0.0));
	vector<vector<complex<double>>> mat2_2d_rand(MAT_SIZE, vector<complex<double>>(MAT_SIZE, 0.0));

	vector<complex<double>> vec1_1d(MAT_SIZE, 0);
	vector<complex<double>> vec2_1d(MAT_SIZE, 0);

	vector<vector<complex<double>>> mat1_2d = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };

	vector<vector<complex<double>>> mat2_2d = { {10, 11, 12}, {13, 14, 15}, {16, 17, 18} };

	for (int row = 0; row < MAT_SIZE; row++)
	{
		for (int col = 0; col < MAT_SIZE; col++)
		{
			int prob = boolean_dist(test_generator);

			//if (prob >= 1)
			//{
			//	mat_2d_rand.at(row).at(col) = 0.0;
			//}

			//else
			//{
			mat1_2d_rand.at(row).at(col) = distribution(test_generator);
			mat2_2d_rand.at(row).at(col) = distribution(test_generator);
			//}


		}
	}

	for (int i = 0; i < MAT_SIZE; i++)
	{
		vec1_1d.at(i) = distribution(test_generator);
		vec2_1d.at(i) = distribution(test_generator);
	}

	int k = 0;
	for (int i = 0; i < MAT_SIZE; i++)
	{
		for (int j = 0; j < MAT_SIZE; j++)
		{
			mat_2d_sq.at(i).at(j) = k;
			k++;
		}
	}

	for (int i = 0; i < MAT_SIZE; i++)
	{
		for (int j = 0; j < MAT_SIZE; j++)
		{
			cout << std::setw(2) << std::setfill('0') << mat_2d_sq.at(i).at(j);
		}
		cout << endl;
	} 

	Operator A(mat1_2d_rand);
	Operator Q(MAT_SIZE, MAT_SIZE);
	Operator R(MAT_SIZE, MAT_SIZE);
	State eig_vec_a1({ 1, 0, 0, 0 });
	Operator A_D(MAT_SIZE, MAT_SIZE);
	Operator A_EIG_VEC(MAT_SIZE, MAT_SIZE);
	Operator A_EIG_VAL(MAT_SIZE, MAT_SIZE);

	Pauli_Matrix sigma_x("Y");

	//A.print();
	//system("PAUSE");

	//time_t start_time = time(NULL);
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	cout << "MATRIX MULTIPLICATION" << endl;

	Operator mat1(mat1_2d_rand);
	Operator mat2(mat2_2d_rand);
	Operator res_mat = mat1 * mat2;

	cout << "SIZE: " << res_mat.get_dim_x() << endl;

	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
	std::cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

	//QR_Algorithm(A, A_EIG_VAL);
	//time_t stop_time = time(NULL);


	//A_EIG_VAL.print();

	//time_t total_time = stop_time - start_time;
	//cout << "TOTAL TIME " << total_time << " SEC" << endl;


	//A.print();
	//system("PAUSE");

	//State psi1(vec1_1d);
	//State psi2(vec2_1d);

	//psi1.print();
	//psi2.print();



	//mat1.print();
	//system("PAUSE");

	//mat2.print();
	//system("PAUSE");

	//time_t start_time = time(NULL);



	//time_t stop_time = time(NULL);



	//time_t total_time = stop_time - start_time;
	//cout << "TOTAL TIME " << total_time << " SEC" << endl;
	//complex<double> calc = inner_product(psi1, psi2);

	//A.transpose();
	//cout << "BEGIN QR ALGORITHM FOR EIGENVALUE DECOMPOSITION" << endl;
	//QR_Algorithm(A, A_EIG_VAL);
	//QR_decomposition(A, Q, R);
	//QR_decomposition(A_D, A_EIG_VEC, A_EIG_VAL);


	//A.print();


	//cout << calc << endl;

	//Operator B(mat1_2d);
	//Operator C(mat2_2d);

	//Operator D = B * C;

	//D.print();



	//A_EIG_VAL.print();
	//R.print();



	//A_EIG_VEC.print();
	//A_EIG_VAL.print();

	//State result_psi = op_rhs(A, eig_vec_a1);

	//result_psi.print();

	//Operator A1(mat_2d_sq);
	//Operator Q(MAT_SIZE, MAT_SIZE);

	//A.print();
	//Hessenberg_reduction(A, Q);
	//Q.print();

	//Operator QT(3, 3);
	//Operator R(3, 3);
	//Operator HESS(3, 3);

	//QR_decomposition(A1, Q, R);

	//QT = Q;
	//QT.transpose();

	//A1.print();
	//Q.print();
	//QT.print();
	//R.print();

	//HESS = QT * A * Q;

	//HESS.print();



	//QR_Algorithm(sigma_x);




	//A1.transpose();

	//A1.print();

	//Operator Q(3, 3);
	//Operator R(3, 3);

	//A.print();

	//QR_decomposition(A1, Q, R);

	//R.print();
	//Q.print();

	//Operator A = Q * R;

	//A.print();

	cout << "ORIGINAL MATRIX A1" << endl;
	A1.print();

	cout << "APPLY GIVENS ROTATION TO ELEMENT (1, 0)" << endl;
	vector<vector<complex<double>>> g1_mat = Givens_rotation(A1.get_dim_x(), A1.get_dim_y(), 1, 0, A1.get_element(0, 0), A1.get_element(1, 0));

	Operator G1(g1_mat);
	cout << "GIVENS ROTATION G1" << endl;
	G1.print();

	cout << "MATRIX MULTIPLICATION: A2 = G1 * A1" << endl;
	Operator A2 = G1 * A1;
	A2.print();

	cout << "APPLY GIVENS ROTATION TO ELEMENT (2, 1)" << endl;
	vector<vector<complex<double>>> g2_mat = Givens_rotation(A2.get_dim_x(), A2.get_dim_y(), 2, 1, A2.get_element(1, 1), A2.get_element(2, 1));

	Operator G2(g2_mat);
	cout << "GIVENS ROTATION G2" << endl;
	G2.print();

	cout << "MATRIX MULTIPLICATION: A3 = G2 * A2" << endl;
	Operator A3 = G2 * A2;
	A3.print(); */
}
void matmult()
{
	std::default_random_engine test_generator;
	std::uniform_int_distribution<int> distribution(0, 100);

	vector<complex<double>> mat1(N * N, 0.0);
	vector<complex<double>> mat2(N * N, 0.0);
	vector<complex<double>> mat3(N * N, 0.0);

	for (unsigned int i = 0; i < N * N; i++)
	{
		mat1[i] = distribution(test_generator);
		mat2[i] = distribution(test_generator);
	}

	cout << "A = " << endl;
	for (unsigned int i = 0; i < mat1.size(); i++)
	{
		//cout << std::setw(2) << std::setfill('0') << mat1[i] << " ";
		if ((i % N) == (N - 1))
		{
			//cout << endl;
		}
	}

	cout << "B = " << endl;
	for (unsigned int i = 0; i < mat2.size(); i++)
	{
		//cout << std::setw(2) << std::setfill('0') << mat2[i] << " ";
		if ((i % N) == (N - 1))
		{
			//cout << endl;
		}
	}

	complex<double> sum = 0.0;
	unsigned int A_index = 0;
	unsigned int B_index = 0;
	unsigned int C_index = 0;

	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	for (unsigned int i = 0; i < N; i++)
	{
		for (unsigned int k = 0; k < N; k++)
		//for (unsigned int j = 0; j < N; j++)
		{
			//sum = 0.0;
			for (unsigned int j = 0; j < N; j++)
			//for (unsigned int k = 0; k < N; k++)
			{
				//A_index = (i * N + k);
				//B_index = (j + k * N);
				
				mat3[(i << n) + j] += (mat1[(i << n) + k] * mat2[j + (k << n)]);
				
				//if (i % 5 == 0 && j % 5 == 0 && k % 5 == 0)
					//cout << "(i, j, k) = (" << i << ", " << j << ", " << k << ")" << endl;
				
				//sum += 
				//sum += (mat1[i * N + k] * mat2[j + k * N]);
				//sum += (mat1[A_index] * mat2[B_index]);
			}
			//C_index = (i * N + j);
			//mat3[C_index] = sum;
			
		}
	}
	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
	std::cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
	

	cout << "C = " << endl;
	for (unsigned int i = 0; i < mat3.size(); i++)
	{
		//cout << std::setw(2) << std::setfill('0') << mat3[i] << " ";
		if ((i % N) == (N - 1))
		{
			//cout << endl;
		}
	}
}
void strassen()
{
	std::default_random_engine test_generator;

	std::uniform_int_distribution<int> distribution(0, 100);

	vector<vector<complex<double>>> mat1_2d_rand(MAT_SIZE, vector<complex<double>>(MAT_SIZE, 0.0));
	vector<vector<complex<double>>> mat2_2d_rand(MAT_SIZE, vector<complex<double>>(MAT_SIZE, 0.0));

	for (int row = 0; row < MAT_SIZE; row++)
	{
		for (int col = 0; col < MAT_SIZE; col++)
		{
			mat1_2d_rand.at(row).at(col) = distribution(test_generator);
			mat2_2d_rand.at(row).at(col) = distribution(test_generator);
		}
	}

	Operator A(mat1_2d_rand);
	Operator B(mat2_2d_rand);
	Operator C(MAT_SIZE, MAT_SIZE);
	Operator A_EIG_VAL(MAT_SIZE, MAT_SIZE);



	cout << "MATRIX MULTIPLICATION" << endl;
	//system("PAUSE");
	//A.print();
	//B.print();
	system("PAUSE");



	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	//Strassen_matrix_multiplication(C, A, B);
	//C = A - B;
	//C = A * B;
	//QR_Algorithm(A, A_EIG_VAL);

	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
	cout << "DONE" << endl;
	//C.print();
	A_EIG_VAL.print();
	std::cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
}

void mat_file_gen()
{
	std::default_random_engine test_generator;

	std::uniform_int_distribution<int> distribution(0, 100);

	std::string file_path = "C:\\Users\\nicks\\Documents\\MAT_TESTS\\";
	std::string file_name = "MAT_";
	
	
	for (int i = 2; i <= NUM_QUBITS; i++)
	{
		for (int counter = 500; counter < 2001; counter++)
		{
			string mat_testname = file_path + file_name + std::to_string(i) + "_" + std::to_string(counter) + ".txt";
			cout << "Generating matric for " << i << " Qubits; Writing to file: " << mat_testname << "..." << endl;

			std::ofstream outfile(mat_testname);
			//vector<vector<complex<double>>> mat_rand(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));

			std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
			for (int j = 0; j < std::pow(2, i); j++)
			{
				for (int k = 0; k < std::pow(2, i); k++)
				{
					//cout << distribution(test_generator) << " ";
					outfile << distribution(test_generator) << " ";
				}
				outfile << endl;
				//cout << endl;
			}
			std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
			cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

			//outfile.close();
		}
	}
	


	

}

void outer_product_test()
{
	std::default_random_engine test_generator;

	std::uniform_int_distribution<int> distribution(0, 100);

	vector<complex<double>> psi1_rand(STATE_SIZE, 0.0);
	vector<complex<double>> psi2_rand(STATE_SIZE, 0.0);

	for (int i = 0; i < STATE_SIZE; i++)
	{
		psi1_rand.at(i) = distribution(test_generator);
		psi2_rand.at(i) = distribution(test_generator);
	}

	State PSI1(psi1_rand);
	State PSI2(psi2_rand);
	Operator MAT(STATE_SIZE, STATE_SIZE);
	complex<double> INNER_PRODUCT = 0.0;

	cout << "STATE SIZE = " << PSI1.get_dim() << endl;
	cout << "OPERATOR SIZE = (" << MAT.get_dim_x() << ", " << MAT.get_dim_y() << ")" << endl;
	system("PAUSE");

	cout << "---> CALCULATING OUTER PRODUCT" << endl;

	std::chrono::steady_clock::time_point outer_product_start_time = std::chrono::steady_clock::now();
	outer_product(MAT, PSI1, PSI2);
	std::chrono::steady_clock::time_point outer_product_stop_time = std::chrono::steady_clock::now();

	cout << "---> CALCULATION COMPLETE\n" << endl;
	
	cout << "---> CALCULATING INNER PRODUCT" << endl;

	std::chrono::steady_clock::time_point inner_product_start_time = std::chrono::steady_clock::now();
	inner_product(INNER_PRODUCT, PSI1, PSI2);
	std::chrono::steady_clock::time_point inner_product_stop_time = std::chrono::steady_clock::now();

	cout << "---> CALCULATION COMPLETE\n" << endl;
		
	cout << "OUTER PRODUCT TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(outer_product_stop_time - outer_product_start_time).count() << " ms" << std::endl;
	cout << "INNER PRODUCT TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(inner_product_stop_time - inner_product_start_time).count() << " ms" << std::endl;
	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(inner_product_stop_time - outer_product_start_time).count() << " ms" << std::endl;
	
}

void hessenberg_test()
{
	std::default_random_engine test_generator;

	std::uniform_int_distribution<int> distribution(0, 100);

	vector<vector<complex<double>>> mat_rand(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));
	//vector<vector<complex<double>>> mat_fixed({ {4, 1, -2, 2}, {1, 2, 0, 1}, {-2, 0, 3, -2}, {2, 1, -2, -1} });
	vector<vector<complex<double>>> mat_fixed({ {2, 2.0 + 1i, 4}, {2.0 - 1i, 3, 1i}, {4, -1i, 1} });

	for (int i = 0; i < STATE_SIZE; i++)
	{
		for (int j = 0; j < STATE_SIZE; j++)
		{
			mat_rand.at(i).at(j) = distribution(test_generator);
		}
	}

	Operator MAT_RAND(mat_rand);
	Operator MAT_FIXED(mat_fixed);
	Operator MAT_HESS = MAT_FIXED;
	Operator MAT_HESS_EIG;

	vector<complex<double>> eigenvalues;

	cout << "OPERATOR SIZE = (" << MAT_RAND.get_dim_x() << ", " << MAT_RAND.get_dim_y() << ")" << endl;
	MAT_HESS.print();
	system("PAUSE");

	cout << std::fpclassify(-0.0) << endl;



	cout << "BEGIN HESSENBERG REDUCTION" << endl;
	//MAT_HESS.print();
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	//Hessenberg_reduction(MAT_HESS);
	cout << "REDUCTION COMPLETE" << endl;

	for (int i = 0; i < MAT_HESS.get_dim_x(); i++)
	{
		for (int j = 0; j < MAT_HESS.get_dim_y(); j++)
		{
			if (std::abs(std::real(MAT_HESS.get_element(i, j))) < 1e-10)
			{
				cout << "0 ";
			}
			else if (std::fpclassify(std::real(MAT_HESS.get_element(i, j))) == FP_NORMAL)
			{
				cout << "* ";
			}
			else
			{
				cout << "? ";
			}
		}
		cout << endl;
	}

	//MAT_HESS.print();
	system("PAUSE");


	QR_Algorithm(MAT_HESS, MAT_HESS_EIG, eigenvalues);
	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

	cout << "HESSENBERG REDUCTION COMPLETE" << endl;

	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;

	cout << "EIGENVALUES" << endl;

	for (int i = 0; i < eigenvalues.size(); i++)
	{
		cout << eigenvalues[i] << endl;
	}

}

void submatrix_test()
{
	std::default_random_engine test_generator;

	std::uniform_int_distribution<int> distribution(0, 100);

	vector<vector<complex<double>>> mat_rand(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));
	vector<vector<complex<double>>> mat_fixed({ {4, 1, -2, 2}, {1, 2, 0, 1}, {-2, 0, 3, -2}, {2, 1, -2, -1} });

	for (int i = 0; i < STATE_SIZE; i++)
	{
		for (int j = 0; j < STATE_SIZE; j++)
		{
			mat_rand.at(i).at(j) = distribution(test_generator);
		}
	}

	Operator MAT_RAND(mat_rand);
	Operator MAT_FIXED(mat_fixed);
	Operator MAT_HESS(STATE_SIZE, STATE_SIZE);

	cout << "OPERATOR SIZE = (" << MAT_FIXED.get_dim_x() << ", " << MAT_FIXED.get_dim_y() << ")" << endl;
	MAT_RAND.print();
	system("PAUSE");

	cout << "BEGIN HESSENBERG REDUCTION" << endl;

	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	//Hessenberg_reduction(MAT_FIXED, MAT_HESS);

	// Get submatrix
	Operator SUB_MAT = MAT_RAND.get_submatrix(0, 3, 0, 3);

	SUB_MAT.print();

	// Double contents
	SUB_MAT *= 2;

	//Overwrite original matrix

	MAT_RAND.set_submatrix(0, 3, 0, 3, SUB_MAT.get_matrix());

	// Print out adjusted matrix

	MAT_RAND.print();

	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

	cout << "HESSENBERG REDUCTION COMPLETE" << endl;

	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
}

/*void gate_test()
{
	Gate1Q PauliX("X");
	Gate1Q PauliY("Y");
	Gate1Q PauliZ("Z");
	Gate1Q Hadamard("H");
	Gate1Q PhaseS("P");
	Gate1Q PI8("T");
	Gate1Q SQRTNOT("SQRT-NOT");
	Gate1Q I2("I");
	Gate1Q Rotation("R", PI / 5);

	State psi0({ 1.0/std::sqrt(2), 1.0/std::sqrt(2) });

	psi0.print();
	Rotation.print();
	State psi1 = op_rhs(Rotation, psi0);

	psi1.print();


}*/

void tensor_product_test()
{
	std::default_random_engine test_generator;

	std::uniform_int_distribution<int> distribution(0, 100);

	vector<vector<complex<double>>> mat1_rand(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));
	vector<vector<complex<double>>> mat2_rand(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));
	vector<vector<complex<double>>> mat1_fixed({ {1, 2}, {3, 4} });
	vector<vector<complex<double>>> mat2_fixed({ {0, 5}, {6, 7} });

	for (int i = 0; i < STATE_SIZE; i++)
	{
		for (int j = 0; j < STATE_SIZE; j++)
		{
			mat1_rand.at(i).at(j) = distribution(test_generator);
			mat2_rand.at(i).at(j) = distribution(test_generator);
		}
	}

	Operator MAT1(mat1_fixed);
	Operator MAT2(mat2_fixed);
	Operator MAT_TP;// (4096, 4096);
	
	MAT1.print();
	system("PAUSE");
	MAT2.print();
	system("PAUSE");

	cout << "BEGIN TENSOR PRODUCT" << endl;
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

	tensor_product(MAT_TP, MAT1, MAT2);

	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

	cout << "TENSOR PRODUCT COMPLETE" << endl;

	MAT_TP.print();

	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
}

void entangle_test()
{
	std::default_random_engine test_generator;

	std::uniform_int_distribution<int> distribution(0, 100);

	vector<complex<double>> vec1_rand(STATE_SIZE, 0.0);
	vector<complex<double>> vec2_rand(STATE_SIZE, 0.0);

	for (int i = 0; i < STATE_SIZE; i++)
	{
		vec1_rand[i] = distribution(test_generator);
		vec2_rand[i] = distribution(test_generator);
		
	}

	vector<complex<double>> in_vec = { 0.7, 0.3 };
	State mypsi(in_vec);
	State psi0(vec1_rand);
	State psi1(vec2_rand);
	State psi_en;

	cout << "BEGIN ENTANGLEMENT" << endl;

	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	entangle(psi_en, psi0, psi1);
	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

	cout << "ENTANGLEMENT COMPLETE" << endl;

	//psi_en.print();
	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
}

void bell_state_test()
{
	//State e1({ 1, 0, 0, 0 });


	State BS1("PHI+");
	State BS2("PHI-");
	State BS3("PSI+");
	State BS4("PSI-");

	BS1.print();
	BS2.print();
	BS3.print();
	BS4.print();
}

void measure_test()
{
	//vector<complex<double>> psi_vec = { 1.0 / 3, 2.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3 };
	
	vector<complex<double>> psi_vec = { 1.0 / 5, 3.0 / 5, 2.0 / 5, 2.0/ 5, 2.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5};
	State psi(psi_vec);

	psi.print();
	system("PAUSE");
	psi.measure();
}

/*
void change_basis_bell_to_standard()
{
	//vector<complex<double>> vec_phi_plus_bs = { 1, 0, 0, 0 };
	//vector<complex<double>> vec_phi_minus_bs = { 0, 1, 0, 0 };
	//vector<complex<double>> vec_psi_plus_bs = { 0, 0, 1, 0 };
	//vector<complex<double>> vec_psi_minus_bs = { 0, 0, 0, 1 };

	vector<complex<double>> vec_phi_plus_bs = { INV_SQRT2, INV_SQRT2, 0, 0 };
	vector<complex<double>> vec_phi_minus_bs = { 0, 0, INV_SQRT2, INV_SQRT2 };
	vector<complex<double>> vec_psi_plus_bs = { 0, 0, INV_SQRT2 , -1.0 * INV_SQRT2 };
	vector<complex<double>> vec_psi_minus_bs = { INV_SQRT2, -1.0 * INV_SQRT2, 0, 0 };

	State phi_plus_bs(vec_phi_plus_bs);
	State phi_minus_bs(vec_phi_minus_bs);
	State psi_plus_bs(vec_psi_plus_bs);
	State psi_minus_bs(vec_psi_minus_bs);

	cout << "BELL BASIS REPRESENTATION OF BELL STATES" << endl;

	phi_plus_bs.print();
	phi_minus_bs.print();

	psi_plus_bs.print();
	psi_minus_bs.print();

	system("PAUSE");

	cout << "CHANGE OF BASIS MATRIX" << endl;

	BELL_TO_STANDARD_BASIS.print();

	system("PAUSE");

	State phi_plus_e = op_rhs(BELL_TO_STANDARD_BASIS, phi_plus_bs);
	State phi_minus_e = op_rhs(BELL_TO_STANDARD_BASIS, phi_minus_bs);
	State psi_plus_e = op_rhs(BELL_TO_STANDARD_BASIS, psi_plus_bs);
	State psi_minus_e = op_rhs(BELL_TO_STANDARD_BASIS, psi_minus_bs);

	cout << "STANDARD BASIS REPRESENTATION OF BELL STATES" << endl;

	phi_plus_e.print();
	phi_minus_e.print();

	psi_plus_e.print();
	psi_minus_e.print();
}

void change_basis_standard_to_bell()
{
	vector<complex<double>> vec_phi_plus_e = { INV_SQRT2, 0, 0, INV_SQRT2 };
	vector<complex<double>> vec_phi_minus_e = { INV_SQRT2, 0, 0, -1.0 * INV_SQRT2 };
	vector<complex<double>> vec_psi_plus_e = { 0, INV_SQRT2, INV_SQRT2, 0 };
	vector<complex<double>> vec_psi_minus_e = { 0, INV_SQRT2, -1.0 * INV_SQRT2, 0 };

	State phi_plus_e(vec_phi_plus_e);
	State phi_minus_e(vec_phi_minus_e);
	State psi_plus_e(vec_psi_plus_e);
	State psi_minus_e(vec_psi_minus_e);

	cout << "STANDARD BASIS REPRESENTATION OF BELL STATES" << endl;

	phi_plus_e.print();
	phi_minus_e.print();

	psi_plus_e.print();
	psi_minus_e.print();

	system("PAUSE");

	cout << "CHANGE OF BASIS MATRIX" << endl;

	STANDARD_TO_BELL_BASIS.print();
	
	system("PAUSE");

	State phi_plus_bs = op_rhs(STANDARD_TO_BELL_BASIS, phi_plus_e);
	State phi_minus_bs = op_rhs(STANDARD_TO_BELL_BASIS, phi_minus_e);
	State psi_plus_bs = op_rhs(STANDARD_TO_BELL_BASIS, psi_plus_e);
	State psi_minus_bs = op_rhs(STANDARD_TO_BELL_BASIS, psi_minus_e);

	cout << "BELL STATE REPRESENTATION OF BELL STATES" << endl;

	phi_plus_bs.print();
	phi_minus_bs.print();

	psi_plus_bs.print();
	psi_minus_bs.print();
}
*/

void quantum_teleportation_test()
{
	vector<complex<double>> unknown_elements{ (3.0 / 5), (4.0 / 5) };

	State psi_c(unknown_elements);

	State psi_ab("PHI+");

	State circuit;

	Gate1Q H_C("H");
	Gate1Q I_B("I");
	Gate2Q I_AB("I");
	Gate2Q CNOT_CA("CNOT");

	H_C.print();
	I_B.print();
	I_AB.print();
	CNOT_CA.print();


	Operator CIRCUIT_OP(8, 8);
	Operator LEVEL1(8, 8);
	Operator LEVEL2(8, 8);
	Operator LEVEL3(8, 8);

	tensor_product(LEVEL1, CNOT_CA, I_B);
	
	LEVEL1.print();
	tensor_product(LEVEL2, H_C, I_AB);
	//tensor_product(LEVEL3, STANDARD_TO_BELL_BASIS, )

	CIRCUIT_OP = LEVEL2 * LEVEL1;

	entangle(circuit, psi_c, psi_ab);

	State EVOLVED_STATE = op_rhs(CIRCUIT_OP, circuit);

	EVOLVED_STATE.print();


	//circuit.print();


}

void test3()
{
	std::default_random_engine test_generator;

	std::uniform_int_distribution<int> distribution(0, 100);

	vector<vector<complex<double>>> mat_rand(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));
	//vector<vector<complex<double>>> mat_fixed({ {4, 1, -2, 2}, {1, 2, 0, 1}, {-2, 0, 3, -2}, {2, 1, -2, -1} });

	for (int i = 0; i < STATE_SIZE; i++)
	{
		for (int j = 0; j < STATE_SIZE; j++)
		{
			complex<double> real_part = distribution(test_generator);
			complex<double> imag_part = distribution(test_generator);
			mat_rand.at(i).at(j) = real_part + imag_part * 1i;
		}
	}

	/*
	//vector<vector<complex<double>>> mat_fixed = { {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16} };
	//vector<vector<complex<double>>> mat_fixed = { {2, 2.0 + 1i, 4}, { 2.0 - 1i, 3, 1i }, { 4, -1i, 1 } };
	//vector<vector<complex<double>>> mat_fixed = {
													{3.0 + 4i, 5.0 + 5i, 1.0 + 1i, 3.0 + 3i, 2.0 + 5i},
													{4.0 + 2i, 1.0 + 4i, 3.0 + 4i, 1.0 + 3i, 4.0 + 3i},
													{2.0 + 2i, 2.0 + 5i, 4.0 + 3i, 3.0 + 2i, 3.0 + 2i},
													{3.0 + 2i, 1.0 + 5i, 2.0 + 2i, 2.0 + 2i, 1.0 + 2i},
													{5.0 + 5i, 3.0 + 2i, 4.0 + 2i, 3.0 + 3i, 3.0 + 3i}
												}; */
	vector<complex<double>> m_eig_A;// {0, 0, 0, 0};

	Operator A(mat_rand);

	Operator Q(STATE_SIZE, STATE_SIZE);
	Operator R(STATE_SIZE, STATE_SIZE);
	Operator EIG_A;

	Operator MAT_HESS = A;
	Operator MAT_HESS_EIG;

	A.print();
	//Hessenberg_reduction(MAT_HESS);
	//MAT_HESS.print();
	system("PAUSE");

	//QR_decomposition(A, Q, R);
	//A.print();
	//Q.print_shape();
	//R.print_shape();
	//R.print();

	//(Q * R).print();

	//A.print();
	system("PAUSE");

	QR_Algorithm(MAT_HESS, EIG_A, m_eig_A);

	//EIG_A.print_shape();
	EIG_A.print();

	for (int i = 0; i < m_eig_A.size(); i++)
	{
		cout << "lambda[" << i << "] = " << m_eig_A[i] << endl;
	}
}

int main()
{
	//matmult();

	//strassen();

	//outer_product_test();

	//hessenberg_test();
	
	//submatrix_test();
	
	//mat_file_gen();
	
	//gate_test();

	//entangle_test();

	//bell_state_test();
	
	//tensor_product_test();

	//measure_test();

	//change_basis_standard_to_bell();

	//change_basis_bell_to_standard();

	//quantum_teleportation_test();

	//vector<complex<double>> in_vector = { 0.6, 0.8 };
	//State UNKNOWN_STATE(in_vector);

	// Declare the Quantum Circuit Object
	//QuantumCircuit QT(3);

	// Initialize the qubits into the required states
	//QT.initialize({ 0 }, UNKNOWN_STATE);
	//QT.initialize({ 1, 2 }, "PHI+");

	// Add the required gates to the circuit
	//QT.add_gate(0, "CNOT", { 0, 1 });
	//QT.add_gate(1, "HADAMARD", { 0 });
	//QT.add_gate(2, "CNOT", { 2, 0 });
	//QT.add_gate(0, "HADAMARD", { 2 });
	//QT.add_gate(5, "HADAMARD", { 0 });
	//QT.add_gate(7, "CNOT", { 1, 2 });

	//QT.entangle("PHI+", { 1, 2 });

	// Evolve the circuit, obtaining the final state at the end of the circuit
	//State result = QT.evolve();

	//QT.print();
	//QT.display();

	vector<vector<complex<double>>> test_vector({ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

	Operator A;
	Operator B(3, 3);
	Operator C(test_vector);
	const Operator D({ {1, 2}, {3, 4} });

	//Operator E({ {0.02, 0.01, 0, 0}, {1, 2, 1, 0}, {0, 1, 2, 1}, {0, 0, 100, 200} });

	std::default_random_engine test_generator;
	std::uniform_int_distribution<int> distribution(0, 100);

	vector<vector<complex<double>>> mat_rand(STATE_SIZE, vector<complex<double>>(STATE_SIZE, 0.0));
	//vector<vector<complex<double>>> mat_fixed({ {4, 1, -2, 2}, {1, 2, 0, 1}, {-2, 0, 3, -2}, {2, 1, -2, -1} });

	for (int i = 0; i < STATE_SIZE; i++)
	{
		for (int j = 0; j < STATE_SIZE; j++)
		{
			//complex<double> real_part = distribution(test_generator);
			//complex<double> imag_part = distribution(test_generator);
			//mat_rand.at(i).at(j) = real_part + imag_part * 1i;
			mat_rand.at(i).at(j) = distribution(test_generator);
		}
	}

	Operator E(mat_rand);

	//E.print();

	//Operator E({ {2, -1, 0}, {-1, 2, -1}, {0, -1, 2} });
	Operator E_EIG(STATE_SIZE, STATE_SIZE);

	Operator IN(STATE_SIZE, STATE_SIZE);
	IN.createIdentityMatrix();

	vector<complex<double>> in_vector(STATE_SIZE, 0.0);

	in_vector[0] = 1.0;

	//E.print();

	vector<complex<double>> eigval_E;

	Operator E_HESS = E;

	Hessenberg_reduction(E_HESS);
	QR_Algorithm(E_HESS, E_EIG, eigval_E);

	for (int i = 0; i < eigval_E.size(); i++)
	{
		cout << "lambda[" << i << "] = " << eigval_E[i] << endl;
	}

	IN *= eigval_E[0];

	Operator DIFF(STATE_SIZE, STATE_SIZE);
	DIFF = E - IN;
	State guess(in_vector);
	State guessk1(in_vector);

	//DIFF.print();

	DIFF.inverse();

	

	cout << "BEING MATRIX INVERSION" << endl;
	system("PAUSE");

	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	//E.inverse();

	for (int i = 0; i < 100; i++)
	{
		guessk1 = op_rhs(DIFF, guess);
		guessk1.normalize();

		guess = guessk1;
	}

	guess.print();

	std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();

	cout << "TOTAL TIME = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() << " ms" << std::endl;
	//E.print();


	return 0;
}