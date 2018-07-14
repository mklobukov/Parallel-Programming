 /* Device code. */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel(float *U, int k)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ double kj_shared[THREAD_BLOCK_SIZE];
	__shared__ double kk_shared;

	// Store elements U[k,k] into shared memory	
	if (threadIdx.x == 0) kk_shared = (double) U[MATRIX_SIZE*k + k];

	__syncthreads();

	if (j > k && j < MATRIX_SIZE){

		__syncthreads();
		
		//DIVISION STEP
		U[MATRIX_SIZE * k + j] = (double) (U[MATRIX_SIZE * k + j] / kk_shared);	

		// Store elements U[k,j] into shared memory
		kj_shared[threadIdx.x] = U[MATRIX_SIZE * k + j];
		
		__syncthreads();

		double temp_kj = kj_shared[j % blockDim.x];

		//ELIMINATION STEP
		for (int i = k+1; i < MATRIX_SIZE; i++){
			
			double temp_ij = U[MATRIX_SIZE * i + j];
			double temp_ik = U[MATRIX_SIZE * i + k];
			
			__syncthreads();
			temp_ij -= __fmul_rn(temp_ik, temp_kj);
			__syncthreads();
			
			U[MATRIX_SIZE * i + j] = temp_ij;
			__syncthreads();
		}	
	}
	
	if (j == MATRIX_SIZE-1){
		U[MATRIX_SIZE * k + k] = 1;
		for (int s = k+1; s < MATRIX_SIZE; s++){
			U[MATRIX_SIZE * s + k] = 0;	
			__syncthreads();	
		}
	}

	__syncthreads();
	
//	if (j == MATRIX_SIZE-1) 
//		U[MATRIX_SIZE * k + k] = 1;

	
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
