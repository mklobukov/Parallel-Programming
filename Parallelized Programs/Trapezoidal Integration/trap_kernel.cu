/* Write GPU kernels to compete the functionality of estimating the integral via the trapezoidal rule. */ 
#include <stdio.h>
#include <math.h>


#define LEFT_ENDPOINT 10
#define RIGHT_ENDPOINT 1005
#define NUM_TRAPEZOIDS 1000000000
#define THREAD_BLOCK_SIZE 256

/* This function uses a compare and swap technique to acquire a mutex/lock. */
__device__ void lock(int *mutex)
{	  
    while(atomicCAS(mutex, 0, 1) != 0);
}

/* This function uses an atomic exchange operation to release the mutex/lock. */
__device__ void unlock(int *mutex)
{
    atomicExch(mutex, 0);
}

__device__ float f_device(float x) {
    return (x + 1)/sqrt(x*x + x + 1);
}  /* f */

__global__ void kernel_trap(float a, float b, int n, float h, double * result, int * mutex) {

	__shared__ float area_per_thread[THREAD_BLOCK_SIZE]; // Allocate shared memory to hold the partial sums.	
	unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // Obtain the thread ID.
	unsigned int stride = blockDim.x * gridDim.x; 
	double sum = 0.0f; 
	unsigned int i = thread_id; 

	/* Compute your partial sum. */
	while(i < n){
		sum += f_device( a + i * (h));
		i += stride;
	}
	
	sum = sum * (h);
	
	area_per_thread[threadIdx.x] = (float)sum; // Copy sum to shared memory.
	__syncthreads(); // Wait for all threads in the thread block to finish up.

	/* Reduce the values generated by the thread block to a single value to be sent back to the CPU. */

	i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i) 
			area_per_thread[threadIdx.x] += area_per_thread[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	/* Accumulate the sum computed by this thread block into the global shared variable. */
	if(threadIdx.x == 0){
		lock(mutex);
		*result += area_per_thread[0];
		unlock(mutex);
	}
}