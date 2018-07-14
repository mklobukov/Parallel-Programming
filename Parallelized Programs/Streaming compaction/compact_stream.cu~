#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <time.h>

// includes, kernels
#include "compact_stream_kernel.cu"

#define NUM_ELEMENTS 1024
//#define THREAD_BLOCK_SIZE 512


void compact_stream(void);
extern "C" unsigned int compare( const float* reference, const float* data, const unsigned int len);
extern "C" void compute_scan_gold( float* reference, float* idata, const unsigned int len);
extern "C" int compact_stream_gold(float *reference, float *idata, unsigned int len);
int compact_stream_on_device(float *result_d, float *h_data, unsigned int num_elements);
int checkResults(float *reference, float *result_d, int num_elements, float threshold);
void printFloatArray(float * array, int num_elements);


int main( int argc, char** argv) 
{
    compact_stream();
    exit(0);
}

void compact_stream(void) 
{
    unsigned int num_elements = NUM_ELEMENTS;
    const unsigned int mem_size = sizeof(float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float *) malloc(mem_size);
      
    // initialize the input data on the host to be integer values
    // between 0 and 1000, both positive and negative
	 srand(time(NULL));
	 float rand_number;
     for( unsigned int i = 0; i < num_elements; ++i) {
         rand_number = rand()/(float)RAND_MAX;
         if(rand_number > 0.5) 
             h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
         else 
             h_data[i] = -floorf(1000*(rand()/(float)RAND_MAX));
     }


    /* Compute reference solution. The function compacts the stream and stores the 
       length of the new steam in num_elements. */
    float *reference = (float *) malloc(mem_size);  
    int stream_length_cpu;
    
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    stream_length_cpu = compact_stream_gold(reference, h_data, num_elements);
    gettimeofday(&stop, NULL);

		printf("Serial Time = %fs. \n", (float)(stop.tv_sec -start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000));

  	/* Add your code to perform the stream compaction on the GPU. 
       Store the result in gpu_result. */
    float *result_d = (float *) malloc(mem_size);
    int stream_length_d;
    stream_length_d = compact_stream_on_device(result_d, h_data, num_elements);


	// Compare the reference solution with the GPU-based solution
    int res = checkResults(reference, result_d, stream_length_cpu, 0.0f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

    // cleanup memory
    free(h_data);
    free(reference);
}

// Use the GPU to compact the h_data stream 
int compact_stream_on_device(float *result_d, float *h_data, unsigned int num_elements)
{
		//allocate h_data on device
		float* d_h_data;
		float * d_result_d;
		int * flag;
		int * scan_output;
		
		cudaMalloc((void**)&d_h_data, num_elements*sizeof(float));
		cudaMalloc((void**)&d_result_d, num_elements*sizeof(float));
    cudaMalloc((void**)&flag, num_elements*sizeof(int));
    cudaMalloc((void**)&scan_output, num_elements*sizeof(int));
    
    
    cudaMemcpy(d_h_data, h_data, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);
    int num_thread_blocks = ceil((float)num_elements/(float)THREAD_BLOCK_SIZE);
    dim3 grid(num_thread_blocks, 1, 1);
    struct timeval start, stop;	
    gettimeofday(&start, NULL);
    compact_stream_kernel<<<grid, thread_block>>>(d_h_data, d_result_d, flag, scan_output, num_elements);
    cudaThreadSynchronize();
    gettimeofday(&stop, NULL);
    printf("Time CUDA = %fs. \n", (float)(stop.tv_sec -start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000));

    
    
    //fix num elements
    cudaMemcpy(result_d, d_result_d, num_elements*sizeof(float), cudaMemcpyDeviceToHost);
    
    /*
    printf("Input: \n");
    printFloatArray(h_data, num_elements);
    
    printf("Result: \n");
    printFloatArray(result_d, num_elements);
    */
    
    
    int n = 0; // Number of elements in the compacted stream

    return n;
}


int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
            break;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}

void printFloatArray(float * array, int num_elements) {
	int i;
	printf("\n\n");
	for (i = 0; i < num_elements; i++) {
		printf("  %.2f  ", array[i]);
	}
	printf("\n\n");
}


