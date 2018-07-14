/*ECEC 622 Final Problem 1
Greg Matthews and Mark Klobukov
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

// includes, kernels
#include "trap_kernel.cu"

double compute_on_device(float, float, int, float);
extern "C" double compute_gold(float, float, int, float);
float function(float );

int 
main(void) 
{
	struct timeval start, stop;
    int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);

	gettimeofday(&start, NULL);
	double reference = compute_gold(a, b, n, h);
	gettimeofday(&stop, NULL);
    printf("Reference solution computed on the CPU = %.9f \n", reference);
  float timeSerial = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
  printf("Time serial: %.5f\n", timeSerial); 

	/* Write this function to complete the trapezoidal on the GPU. */
	double gpu_result = compute_on_device(a, b, n, h);
	printf("Solution computed on the GPU = %.9f \n", gpu_result);
	printf("Difference: %.6f\n", reference-gpu_result);
} 

float function(float x) {
    return (x + 1)/sqrt(x*x + x + 1);
}  /* f */



/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_on_device(float a, float b, int n, float h)
{
		struct timeval start, stop;
		double * result = NULL;
		double sum; 

		cudaMalloc((void**)&result, sizeof(double));
		cudaMemset(result, 0.0f, sizeof(double));

		int *mutex_on_device = NULL;
		cudaMalloc((void **)&mutex_on_device, sizeof(int));
		cudaMemset(mutex_on_device, 0, sizeof(int));
		
		dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);
		dim3 grid (n/THREAD_BLOCK_SIZE, 1, 1);
		
		gettimeofday(&start, NULL);
		kernel_trap<<<grid, thread_block>>>(a, b, n, h, result, mutex_on_device);
		cudaThreadSynchronize();
		gettimeofday(&stop, NULL);
		float timeParallel = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
		printf("Time parallel: %.5f\n", timeParallel);
		
		cudaMemcpy(&sum, result, sizeof(double), cudaMemcpyDeviceToHost);
		
		sum = sum - h/2* (function(a) + function(b) );
		
		//Free Memory on Device
		cudaFree(result);
		result = NULL;
		
    return sum;
}



