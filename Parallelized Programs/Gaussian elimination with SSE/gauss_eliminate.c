/* Gaussian elimination code.
 * Author: Naga Kandasamy, 10/24/2015
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -std=c99 -O3 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "gauss_eliminate.h"
#include <xmmintrin.h>


#define MIN_NUMBER 2
#define MAX_NUMBER 50

extern int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void gauss_eliminate_using_sse(float*, const float*, unsigned int);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, unsigned int, float);
void printMatrix(const Matrix);

int main(int argc, char** argv) {
    if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}	

    /* Allocate and initialize the matrices. */
	Matrix  A;                                              /* The N x N input matrix. */
	Matrix  U;                                              /* The upper triangular matrix to be computed. */
	
	srand(time(NULL));
		
    A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);      /* Create a random N x N matrix. */
	U  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);      /* Create a random N x 1 vector. */
		
	/* Gaussian elimination using the reference code. */
	Matrix reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	struct timeval start, stop;	
	gettimeofday(&start, NULL);

	printf("Performing gaussian elimination using the reference code. \n");
	int status = compute_gold(reference.elements, A.elements, A.num_rows);

	gettimeofday(&stop, NULL);
	printf("CPU run time = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	if(status == 0){
		printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
		exit(0);
	}
	status = perform_simple_check(reference); // Check that the principal diagonal elements are 1 
	if(status == 0){
		printf("The upper triangular matrix is incorrect. Exiting. \n");
		exit(0); 
	}
	printf("Gaussian elimination using the reference code was successful. \n");

		

	/* WRITE THIS CODE: Perform the Gaussian elimination using the SSE version. 
     * The resulting upper triangular matrix should be returned in U
     * */
	
	gettimeofday(&start, NULL);
	gauss_eliminate_using_sse(U.elements, A.elements, U.num_rows);
	gettimeofday(&stop, NULL);

	printf("SSE run time = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	//printMatrix(reference);
	//printf("\n");
	//printMatrix(U);
	
	/* Check if the SSE result is equivalent to the expected solution. */
	int size = MATRIX_SIZE*MATRIX_SIZE;
	int res = check_results(reference.elements, U.elements, size, 0.001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	free(A.elements); A.elements = NULL;
	free(U.elements); U.elements = NULL;
	free(reference.elements); reference.elements = NULL;

	return 0;
}


void 
gauss_eliminate_using_sse(float* U, const float* A, unsigned int num_elements)                  /* Write code to perform gaussian elimination using OpenMP. */
{
		unsigned int i, j, k, init;
		float tmp[4] __attribute__ ((aligned(16)));

		__m128 m0, m1, m2, m3;
	// INITIALIZATION
    for (i = 0; i < num_elements; i ++)             /* Copy the contents of the A matrix into the U matrix. */
        for(j = 0; j < num_elements/ 4; j++){
			m0 = _mm_load_ps(&A[num_elements*i + 4*j]);
			_mm_store_ps(&U[num_elements*i + 4*j], m0);
		}
		

    for (k = 0; k < num_elements; k++){
        
		m0 = _mm_set_ps1(U[num_elements*k + k]);	
		init = k+1;

		// INITIAL SERIAL DIVISION
		while ((init % 4) != 0){	
			U[num_elements * k + init] = (U[num_elements * k + init] / U[num_elements * k + k]);
			init++; 
		}
	
		// SSE DIVISION STEP	
		for (j = init; j < num_elements; j+=4){   /* Reduce the current row. */
	
			m1 = _mm_load_ps(&U[num_elements*k + j]);
			m1 = _mm_div_ps(m1,m0);
			_mm_store_ps(&U[num_elements*k + j], m1);
        }
	
		U[num_elements * k + k] = 1;             /* Set the principal diagonal entry in U to be 1. */

	
		// ELIMINATION STEP
        for (i = (k+1); i < num_elements; i++){
			
			init = k+1;	

				// INITAL SERIAL ELIMINATION
				while ((init % 4) != 0){
					U[num_elements * i + init] = U[num_elements * i + init] - (U[num_elements * i + k] * U[num_elements * k + init]);
					init++;
				}
			
				m0 = _mm_load1_ps(&U[num_elements*i + k]);
			
				// SSE ELIMINATION	
				for (j = init; j < num_elements; j+=4){
				
					m2 = _mm_load_ps(&U[num_elements*i + j]);
					m3 = _mm_load_ps(&U[num_elements*k + j]);
					m3 = _mm_mul_ps(m3, m0);
					m2 = _mm_sub_ps(m2, m3);
					_mm_store_ps(&U[num_elements*i + j], m2);
				}

				U[num_elements * i + k] = 0; 
		}
	}	
}


int 
check_results(float *A, float *B, unsigned int size, float tolerance)   /* Check if refernce results match multi threaded results. */
{
	for(int i = 0; i < size; i++)
		if(fabsf(A[i] - B[i]) > tolerance)
			return 0;
	
    return 1;
}


/* Allocate a matrix of dimensions height*width. 
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization.
 * */
Matrix 
allocate_matrix(int num_rows, int num_columns, int init){
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

	void* allocation;
	posix_memalign(&allocation, 16, sizeof(float) * size);
	
	M.elements = (float*) allocation;

    for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}

    return M;
}	


float 
get_random_number(int min, int max){                                    /* Returns a random FP number between min and max values. */
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

int 
perform_simple_check(const Matrix M){                                   /* Check for upper triangular matrix, that is, the principal diagonal elements are 1. */
    for(unsigned int i = 0; i < M.num_rows; i++)
        if((fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) return 0;
	
    return 1;
} 

void printMatrix(const Matrix M){
	for (unsigned int i = 0; i < M.num_rows; i++) {
		for (unsigned int j = 0; j < M.num_rows; j++) {
			printf("%.3f ", M.elements[M.num_rows*i + j]);
		}
		printf("\n");
	}
}


