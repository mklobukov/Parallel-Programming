/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

/* Write the kernel for vector-matrix multiplication using GPU global memory. */
__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	// Find the positions in Matrix
	
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	double Y_temp = 0.0;
	
	for( int k = 0; k < MATRIX_SIZE; k++)
	{
		double A_element = Ad[MATRIX_SIZE * col + k]; // Scan through row elements
		double X_element = Xd[k];
		Y_temp += A_element * X_element; 
	}
	
	Yd[col] = (float)Y_temp;

}


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	//Locate the thread in output vector
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	//Shared portion of X
	__shared__ float X_shared[BLOCK_DIM_X];
	
	double Y_temp = 0.0;
	double X_element;
	double A_element;
	
	for (int m = 0; m < MATRIX_SIZE / BLOCK_DIM_X; m++) {
		X_shared[threadIdx.x] = Xd[m*BLOCK_DIM_X + threadIdx.x];
		__syncthreads();
		for( int k = BLOCK_DIM_X*m; k < BLOCK_DIM_X * (m+1); k++)
		{
			A_element = Ad[MATRIX_SIZE * col + k]; 
			X_element = X_shared[k % BLOCK_DIM_X];
			Y_temp += A_element * X_element; 
		}
		__syncthreads();
	}
	Yd[col] = (float)Y_temp;
}
	



#endif // #ifndef _MATRIXMUL_KERNEL_H_
