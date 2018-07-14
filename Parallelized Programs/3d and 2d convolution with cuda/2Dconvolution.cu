/* 2D convolution for ECEC 622
* Code skeleton provided by Dr. Kandasamy
* Parallel computation modifications added by
* Greg Matthews and Mark Klobukov
* 3/16/2017
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define KERNEL_SIZE 5
// includes, kernels
__constant__ float kernel_c[KERNEL_SIZE*KERNEL_SIZE]; // Allocation for the kernel in GPU constant memory
#include "2Dconvolution_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

void printMatrix(const Matrix);
void check_for_error(const char *);
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P_global, Matrix P_shared);
int checkResults(float *, float *, int, float);

int main(int argc, char** argv) 
{

	Matrix  A;
	Matrix  B;
	Matrix  P_global;
	Matrix P_shared;
	
	srand(time(NULL));
	
	// Allocate and initialize the matrices
	A  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
	B  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	P_global  = AllocateMatrix(B.height, B.width, 0);
	P_shared = AllocateMatrix(B.height, B.width, 0);
    
   /* Convolve matrix B with matrix A on the CPU. */
   Matrix reference = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
    struct timeval start, stop;	
		gettimeofday(&start, NULL);
   computeGold(reference.elements, A.elements, B.elements, B.height, B.width);
   	gettimeofday(&stop, NULL);
   	printf("CPU Run Time = %fs. \n", (float)(stop.tv_sec - start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000));
   	
   
       
	/* Convolve matrix B with matrix A on the device. */
   ConvolutionOnDevice(A, B, P_global, P_shared);

   

   /* Check if the device result is equivalent to the expected solution. */
    int num_elements = P_global.height * P_global.width;
	int status = checkResults(reference.elements, P_global.elements, num_elements, 0.001f);
	printf("Test global: %s\n", (1 == status) ? "PASSED" : "FAILED"); 

		status = checkResults(reference.elements, P_shared.elements, num_elements, 0.001f);
			printf("Test shared: %s\n", (1 == status) ? "PASSED" : "FAILED"); 	
	
	
	

   // Free matrices
   FreeMatrix(&A);
   FreeMatrix(&B);

	
   return 0;
}


void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P_global, Matrix P_shared)
{
    // Load M and N to the device, time the CPU-GPU communication overhead
    struct timeval start, stop; 
    Matrix Md = AllocateDeviceMatrix(M);
    gettimeofday(&start, NULL);
    CopyToDeviceMatrix(Md, M);
    gettimeofday(&stop, NULL);
    float timeForMd = (float)(stop.tv_sec \
	 -start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000);
    gettimeofday(&start, NULL);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd_global = AllocateDeviceMatrix(P_global);
    CopyToDeviceMatrix(Pd_global, P_global); // Clear memory

	Matrix Pd_shared = AllocateDeviceMatrix(P_shared);
	CopyToDeviceMatrix(Pd_shared, P_shared); // Clear memory

    // Setup the execution configuration
    int num_elements = N.width;
    dim3 thread_block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1);
    int num_thread_blocks = num_elements/THREAD_BLOCK_SIZE;
    dim3 grid(num_thread_blocks, num_thread_blocks, 1);
    
    gettimeofday(&stop, NULL);
    float overheadTime = (float)(stop.tv_sec \
	 -start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000);

    
    // Launch the device computation threads!
    //struct timeval start, stop;	
	
	gettimeofday(&start, NULL);
    ConvolutionKernel<<<grid, thread_block>>>(Md, Nd, Pd_global);
    cudaThreadSynchronize();
    gettimeofday(&stop, NULL);
    
	printf("Global Memory Time = %fs. \n", (float)(stop.tv_sec \
	 -start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000));
    
	 // Read P from the device
	  gettimeofday(&start, NULL);
    CopyFromDeviceMatrix(P_global, Pd_global);
    FreeDeviceMatrix(&Pd_global);
    gettimeofday(&stop, NULL);
    overheadTime += (float)(stop.tv_sec \
	 -start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000);
    check_for_error("KERNEL FAILURE");
    
	// We copy the mask to GPU constant memory in an attempt to improve the performance
	gettimeofday(&start, NULL);
	cudaMemcpyToSymbol(kernel_c, M.elements, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
 gettimeofday(&stop, NULL);
 float constMemOverhead = (float)(stop.tv_sec \
	 -start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000);
    // Launch the device computation threads!
	gettimeofday(&start, NULL);
    ConvolutionKernel_optimized<<<grid, thread_block>>>(Nd, Pd_shared);
    cudaThreadSynchronize();
    gettimeofday(&stop, NULL);
    printf("Optimized CUDA time = %fs. \n", (float)(stop.tv_sec \
	-start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000));
		
    // Read P from the device
    CopyFromDeviceMatrix(P_shared, Pd_shared);

		gettimeofday(&start, NULL);
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd_shared);
		gettimeofday(&stop, NULL);
		overheadTime += (float)(stop.tv_sec \
	 -start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000);
	 printf("Naive approach overhead: %.5f\n", (float)(overheadTime + timeForMd));
	 printf("Optimized approach overhead: %.5f\n", (float)(overheadTime + constMemOverhead));
		
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++){
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Check the CPU and GPU solutions
int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}

void 
check_for_error(const char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


void printMatrix(const Matrix M){
	for (unsigned int i = 0; i < M.width; i++) {
		for (unsigned int j = 0; j < M.width; j++) {
			printf("%.3f ", M.elements[M.width*i + j]);
		}
		printf("\n");
	}
}
