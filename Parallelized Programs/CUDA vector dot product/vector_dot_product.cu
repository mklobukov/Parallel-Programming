/*
 ECEC 622 Vector Dot Product CUDA implementation
 Skeleton code author: Dr. Kandasamy
 Greg Matthews and Mark Klobukov
 3/8/2017
*/



#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, kernels
#include "vector_dot_product_kernel.cu"

void run_test(unsigned int);
float compute_on_device(float *, float *,int);
void check_for_error(char *);
extern "C" float compute_gold( float *, float *, unsigned int);

int main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: vector_dot_product <num elements> \n");
		exit(0);	
	}
	unsigned int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

/* Perform vector dot product on the CPU and the GPU and compare results for correctness.  */
void run_test(unsigned int num_elements) 
{
	struct timeval start, stop;
	// Obtain the vector length
	unsigned int size = sizeof(float) * num_elements;

	// Allocate memory on the CPU for the input vectors A and B
	float *A = (float *)malloc(size);
	float *B = (float *)malloc(size);
	
	// Randomly generate input data. Initialize the input data to be floating point values between [-.5 , 5]
	printf("Generating random vectors with values between [-.5, .5]. \n");	
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;
		B[i] = (float)rand()/(float)RAND_MAX - 0.5;
	}
	
	printf("Generating dot product on the CPU. \n");
	
	// Compute the reference solution on the CPU
	gettimeofday(&start, NULL);
	float reference = compute_gold(A, B, num_elements);
	gettimeofday(&stop, NULL);
	float timeCPU = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("CPU run time: %.5f\n", timeCPU);
	

	/* Edit this function to compute the result vector on the GPU. 
       The result should be placed in the gpu_result variable. */
	float gpu_result = compute_on_device(A, B, num_elements);

    /* Compare the CPU and GPU results. */
    float threshold = 0.001;
	printf("Result on CPU: %f, result on GPU: %f. \n", reference, gpu_result);
    if(fabsf((reference - gpu_result)/reference) < threshold){
        printf("TEST passed. \n");
    }
    else{
        printf("TEST failed. \n");
    }

	// cleanup memory
	free(A);
	free(B);
	
	return;
}

/* Edit this function to compute the dot product on the device. */
float compute_on_device(float *A_on_host, float *B_on_host, int num_elements)
{
		float *A_on_device = NULL;
		float *B_on_device = NULL;
		float *result_on_device = NULL;
    float result = 0;
    struct timeval start, stop;
    
    	/* Allocate space on the GPU for vector A and copy the contents to the GPU. */
	cudaMalloc((void**)&A_on_device, num_elements * sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	
		/* Allocate space on the GPU for vector B and copy the contents to the GPU. */
	cudaMalloc((void**)&B_on_device, num_elements * sizeof(float));
	cudaMemcpy(B_on_device, B_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	
	/* Allocate space for the result on the GPU and initialize it. */
	cudaMalloc((void**)&result_on_device, sizeof(float));
	cudaMemset(result_on_device, 0.0f, sizeof(float));
	
	/* Allocate space for the lock on the GPU and initialize it. */
	int *mutex_on_device = NULL;
	cudaMalloc((void **)&mutex_on_device, sizeof(int));
	cudaMemset(mutex_on_device, 0, sizeof(int));
	
		/* Set up the execution grid on the GPU. */
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); 
	dim3 grid(NUM_BLOCKS,1);
	
	gettimeofday(&start, NULL);
		/* Launch the kernel. */
	vector_dot_product_kernel<<<grid, thread_block>>>(A_on_device, B_on_device, result_on_device, num_elements, mutex_on_device);
	cudaThreadSynchronize();
	gettimeofday(&stop, NULL);
	float timeCUDA = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("CUDA run time: %.5f\n", timeCUDA);
    
  cudaMemcpy(&result, result_on_device, sizeof(float), cudaMemcpyDeviceToHost);  
    
  /* Free memory. */
	cudaFree(A_on_device);
	cudaFree(B_on_device);
	cudaFree(result_on_device); 
  cudaFree(mutex_on_device);
  return result;
}
 
/* This function checks for errors returned by the CUDA run time. */
void check_for_error(char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
