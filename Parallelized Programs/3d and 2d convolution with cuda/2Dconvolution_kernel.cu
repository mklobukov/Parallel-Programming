
#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int wN = N.width;
	int hN = N.height;
	
	if (i >= N.height || j >= N.width) return;
	
	// For each element in the result  matrix
	double sum = 0;
	// check the start and end values of m and n to prevent 
	// overrunning the matrix edges
	unsigned int mbegin = (i < 2)? 2 - i : 0;
	unsigned int mend = (i > (hN - 3))?
							hN - i + 2 : 5;
	unsigned int nbegin = (j < 2)? 2 - j : 0;
	unsigned int nend = (j > (wN - 3))?
							(wN-j) + 2 : 5;
	
	// overlay M over N centered at element (i,j).  For each 
	//  overlapping element, multiply the two and accumulate
	for(unsigned int m = mbegin; m < mend; ++m) {
		for(unsigned int n = nbegin; n < nend; n++) {
			sum += M.elements[m * 5 + n] * 
					N.elements[wN*(i + m - 2) + (j+n - 2)];
		}
	}
	// store the result
	P.elements[i*wN + j] = (float)sum;

}

__global__ void ConvolutionKernel_optimized(Matrix N, Matrix P) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	__shared__ float N_ds[THREAD_BLOCK_SIZE*THREAD_BLOCK_SIZE];
	N_ds[THREAD_BLOCK_SIZE * threadIdx.y + threadIdx.x] = N.elements[N.width * y + x];
	__syncthreads();
	
	int this_start_x = blockIdx.x*blockDim.x;
	int this_start_y = blockIdx.y*blockDim.y;
	int next_start_x = (blockIdx.x+1)*blockDim.x;
	int next_start_y = (blockIdx.y+1)*blockDim.y;
	
	int N_start_x = x - (KERNEL_SIZE/2);
	int N_start_y = y - (KERNEL_SIZE/2);
	
	double Pvalue = 0;
	
	for (int k = 0; k < KERNEL_SIZE; k++) {
		int N_index_y = N_start_y + k;
		if (N_index_y >= 0 && N_index_y < N.width) {
			for (int r = 0; r < KERNEL_SIZE; r++) {
				int N_index_x = N_start_x + r;
				if (N_index_x >= 0 && N_index_x < N.width) {
					if ((N_index_x >= this_start_x) 
							 	&& (N_index_y >= this_start_y) 
					 			&& (N_index_x < next_start_x)
					 			&& (N_index_y < next_start_y) ) {
				Pvalue += N_ds[THREAD_BLOCK_SIZE * (threadIdx.y + k - KERNEL_SIZE/2) \
				+ threadIdx.x + r - (KERNEL_SIZE/2)] * kernel_c[KERNEL_SIZE * k + r];
					} 
					else {
				Pvalue += N.elements[N.width * N_index_y + N_index_x] \
				* kernel_c[KERNEL_SIZE * k + r];
					}
				}
			}
		}
	}
	P.elements[P.width * y + x] = (float)Pvalue;
}


#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
