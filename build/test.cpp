
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using std::cout;
using std::endl;

__host__ __device__ int sum(int a, int b)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0)
	printf("I AM ON DEVICE\n");
#else
	printf("I AM ON HOST\n");
#endif

	return a + b;
}

int main()
{
#if defined(USE_GPU)
		cout << "USING GPU CODE" << endl;
#else
		cout << "USING CPU CODE" << endl;
#endif

	
#ifdef __CUDACC__
	cout << "NVCC USED AS COMPILER" << endl;
#else
	cout << "GCC USED AS COMPILER" << endl;
#endif

	cout << "SALVE MUNDI" << endl;
	
	//int total = sum(2, 2);


	return 0;
}


