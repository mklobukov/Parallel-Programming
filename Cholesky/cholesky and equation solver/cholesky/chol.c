/* Cholesky decomposition.
* Serial implementation provided by Dr. Kandasamy
* Parallel versions of the algorithm written by
* Mark Klobukov and Gregory Matthews for
* the ECEC622 Midterm
*2/18/2017
 * Compile as follows:
 * gcc -fopenmp -o chol chol.c chol_gold.c -pthread -lm -std=gnu99
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "chol.h"
#include <sys/time.h>
#include <pthread.h>

#define NUM_THREADS 16

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

typedef struct barrier_struct {
	pthread_mutex_t mutex; //protects access to the value
	pthread_cond_t condition; //signals change to the value
	int counter; //the value
} barrier_t;

barrier_t barrier = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
barrier_t barrier2 = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
barrier_t barrier3 = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};

//data struture for passing arguments for each worker thread
typedef struct args_for_thread_s {
	int thread_id; //thread id
	int num_elements; //number of rows/cols n in the nxn matrix
	float *matrixA; //starting address for matrix A
	float *matrixU; //starting address for matrix U
	int num_rows; //number of rows on which a given thread will operate
} ARGS_FOR_THREAD;

Matrix allocate_matrix(int num_rows, int num_columns, int init);
int perform_simple_check(const Matrix M);
void print_matrix(const Matrix M);
extern Matrix create_positive_definite_matrix(unsigned int, unsigned int);
extern int chol_gold(const Matrix, Matrix);
extern int check_chol(const Matrix, const Matrix);
void chol_using_pthreads(float *, float*, unsigned int);
int chol_using_openmp(const Matrix, Matrix);
void * chol_pthread(void *);
void barrier_sync(barrier_t *);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 struct timeval start, stop; //for measuring time
	// Matrices for the program
	Matrix A; // The N x N input matrix
	Matrix reference; // The upper triangular matrix computed by the CPU
	Matrix U_pthreads; // The upper triangular matrix computed by the pthread implementation
	Matrix U_openmp; // The upper triangular matrix computed by the openmp implementation 
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));

	// Create the positive definite matrix. May require a few tries if we are unlucky
	int success = 0;
	while(!success){
		A = create_positive_definite_matrix(MATRIX_SIZE, MATRIX_SIZE);
		if(A.elements != NULL)
				  success = 1;
	}
	
	// getchar();


	reference  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the CPU result
	U_pthreads =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the pthread result
	U_openmp =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the openmp result


	// compute the Cholesky decomposition on the CPU; single threaded version	
	printf("Performing Cholesky decomposition on the CPU using the single-threaded version. \n");
	gettimeofday(&start, NULL);
	int status = chol_gold(A, reference);
	gettimeofday(&stop, NULL);
	if(status == 0){
			  printf("Cholesky decomposition failed. The input matrix is not positive definite. \n");
			  exit(0);
	}
	float serialTime = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("Serial program run time = %.4f s. \n", serialTime);
	
	/*
	printf("Double checking for correctness by recovering the original matrix. \n");
	if(check_chol(A, reference) == 0){
		printf("Error performing Cholesky decomposition on the CPU. Try again. Exiting. \n");
		exit(0);
	}
	*/

	printf("Cholesky decomposition on the CPU was successful. \n");

	/* MODIFY THIS CODE: Perform the Cholesky decomposition using pthreads. The resulting upper triangular matrix should be returned in 
	 U_pthreads */

	gettimeofday(&start, NULL);
	chol_using_pthreads(A.elements, U_pthreads.elements, A.num_rows);
	gettimeofday(&stop, NULL);
	float pthreadTime = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("Pthread time = %.4f\n", pthreadTime);
	printf("PThreads Speedup = %.4f \n", serialTime/pthreadTime);	


	/* MODIFY THIS CODE: Perform the Cholesky decomposition using openmp. The resulting upper traingular matrix should be returned in U_openmp */
	gettimeofday(&start, NULL);
	chol_using_openmp(A, U_openmp);
	gettimeofday(&stop, NULL);
	float openmpTime = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("OpenMP run time = %.4f s. \n", openmpTime);
	printf("OpenMP Speedup = %.4f \n", serialTime/openmpTime);


	// Check if the pthread and openmp results are equivalent to the expected solution
	if(check_chol(A, U_pthreads) == 0) 
			  printf("Error performing Cholesky decomposition using pthreads. \n");
	else
			  printf("Cholesky decomposition using pthreads was successful. \n");

	if(check_chol(A, U_openmp) == 0) 
			  printf("Error performing Cholesky decomposition using openmp. \n");
	else	
			  printf("Cholesky decomposition using openmp was successful. \n");

	
	//print_matrix(reference);

	//printf("\n");

	//print_matrix(U_openmp);

	//printf("\n");

	//print_matrix(U_pthreads);



	// Free host matrices
	free(A.elements); 	
	free(U_pthreads.elements);	
	free(U_openmp.elements);
	free(reference.elements); 
	return 1;
}

/* Write code to perform Cholesky decopmposition using pthreads. */
void chol_using_pthreads(float* A, float* U, unsigned int num_elements)
{
	unsigned int size = num_elements * num_elements;
	unsigned int i, k;
	for (i = 0; i < size; i ++)
		U[i] = A[i];
	
	pthread_t thread_id[NUM_THREADS]; //data structure to store thread ID
	pthread_attr_t attributes; //Thread attributes
	pthread_attr_init(&attributes); //initialize thread attributes to default
	ARGS_FOR_THREAD * args[NUM_THREADS];
	
	for(i = 0; i < NUM_THREADS; i++) {
		args[i] = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
		args[i]->thread_id = i; //thread ID
		args[i]-> num_elements = num_elements;
		args[i]->matrixA = A;//can g oaway
		args[i]->matrixU = U;
	}
	

	for (i = 0; i < NUM_THREADS; i++) {
		pthread_create(&thread_id[i], &attributes, chol_pthread, (void*) args[i]);
	}

	//DO THE JOIN HERE
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread_id[i], NULL);
	}

	
	//Free memory used by the threads
	
	for (i = 0; i < NUM_THREADS; i++) 
		free((void*)args[i]);
}

void * chol_pthread(void * args) {

ARGS_FOR_THREAD * my_args = (ARGS_FOR_THREAD*)args;
unsigned int i, j, k; 
int num_elements = my_args->num_elements;
unsigned int firstIndex, lastIndex, chunk, offset;
float * U = my_args->matrixU;


for (k = 0; k < num_elements; k++) {
	// Perform the Cholesky decomposition in place on the U matrix
	chunk = (int)floor((float)(num_elements - k) / (float) NUM_THREADS); //recalculate chunk size for each iteration
	offset = my_args->thread_id*chunk;
	firstIndex = k + 1 + my_args->thread_id*chunk;
	lastIndex= firstIndex + chunk;
	
	if (my_args->thread_id == (NUM_THREADS-1)) {
		lastIndex = num_elements;
	}
			  // Take the square root of the diagonal elemen
			  if (my_args->thread_id == 0)
				U[k * num_elements + k] = sqrt(U[k * num_elements + k]);
			  
			  if(U[k * num_elements + k] <= 0) {
					printf("Cholesky decomposition failed. \n");
					exit(0); }
			 

			  barrier_sync(&barrier);

			  // Division step
			  for(j = firstIndex; j < lastIndex; j++)
						 U[k * num_elements + j] /= U[k * num_elements + k]; // Division step


				barrier_sync(&barrier2);
				
			  // Elimination step
			  for(i = firstIndex; i < lastIndex; i++)
						 for(j = i; j < num_elements; j++)
									U[i * num_elements + j] -= U[k * num_elements + i] * U[k * num_elements + j];

			  barrier_sync(&barrier3);
	}

	// As the final step, zero out the lower triangular portion of U
	for(i = 0; i < num_elements; i++)
			  for(j = 0; j < i; j++)
						 U[i * num_elements + j] = 0.0;



}



/* Write code to perform Cholesky decopmposition using openmp. */
int chol_using_openmp(const Matrix A, Matrix U)
{

unsigned int i, j, k; 
	unsigned int size = A.num_rows * A.num_columns;

	// Copy the contents of the A matrix into the working matrix U
# pragma omp parallel for	num_threads(NUM_THREADS) default(none) private(i) shared(U, size)
	for (i = 0; i < size; i ++)
		U.elements[i] = A.elements[i];

	// Perform the Cholesky decomposition in place on the U matrix
	for(k = 0; k < U.num_rows; k++){
			  // Take the square root of the diagonal element
			  U.elements[k * U.num_rows + k] = sqrt(U.elements[k * U.num_rows + k]);
			  if(U.elements[k * U.num_rows + k] <= 0){
						 printf("Cholesky decomposition failed. \n");
						 return 0;
			  }
			  
# pragma omp parallel for num_threads(NUM_THREADS)
			  // Division step
			  for(j = (k + 1); j < U.num_rows; j++)
						 U.elements[k * U.num_rows + j] /= U.elements[k * U.num_rows + k]; // Division step

# pragma omp parallel for num_threads(NUM_THREADS) default(none) private(i,j) shared(U,k)
			  // Elimination step
			  for(i = (k + 1); i < U.num_rows; i++)
						 for(j = i; j < U.num_rows; j++)
									U.elements[i * U.num_rows + j] -= U.elements[k * U.num_rows + i] * U.elements[k * U.num_rows + j];

	}

	// As the final step, zero out the lower triangular portion of U
	for(i = 0; i < U.num_rows; i++)
			  for(j = 0; j < i; j++)
						 U.elements[i * U.num_rows + j] = 0.0;

	// printf("The Upper triangular matrix is: \n");
	// print_matrix(U);
	return 1;
}


// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float *) malloc(size * sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = (float)rand()/(float)RAND_MAX;
	}
    return M;
}	

/* The function that implements the barrier synchronization. */
void 
barrier_sync(barrier_t *barrier)
{
    pthread_mutex_lock(&(barrier->mutex));
    barrier->counter++;
   // printf("Barrier at %d\n", barrier->counter);
    /* Check if all threads have reached this point. */
    if(barrier->counter == NUM_THREADS){
        barrier->counter = 0; // Reset the counter			 
        pthread_cond_broadcast(&(barrier->condition)); /* Signal this condition to all the blocked threads. */
    } 
    else{
        /* We may be woken up by events other than a broadcast. If so, we go back to sleep. */
        while((pthread_cond_wait(&(barrier->condition), &(barrier->mutex))) != 0); 		  
    }
		 
    pthread_mutex_unlock(&(barrier->mutex));
}


