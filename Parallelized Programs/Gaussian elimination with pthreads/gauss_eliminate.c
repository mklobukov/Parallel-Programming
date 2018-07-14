/* Gaussian elimination code
 * Author: Naga Kandasamy
 * Date created: 02/07/2014
 * Date of last update: 01/30/2017
 * Modified by Greg Matthews and Mark Klobukov
 * 2/11/2017
 * Compile as follows: gcc -pthread -o gauss_eliminate gauss_eliminate.c compute_gold.c -std=gnu99 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define NUM_THREADS 32
//#define _GNU_SOURCE
int flag = 0;
//define a data structure to synchronize threads
typedef struct barrier_struct {
	pthread_mutex_t mutex; //protects access to the value
	pthread_cond_t condition; //signals change to the value
	int counter; //the value
} barrier_t;

barrier_t barrier = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
barrier_t barrier2 = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};

//data struture for passing arguments for each worker thread
typedef struct args_for_thread_s {
	int thread_id; //thread id
	int num_elements; //number of rows/cols n in the nxn matrix
	float *matrixA; //starting address for matrix A
	float *matrixU; //starting address for matrix U
	int offset; 
	int num_rows; //number of rows on which a given thread will operate
} ARGS_FOR_THREAD;


//pthread_barrier_t barrier, barrier2; //barrier synchronization object

/* Function prototypes. */
void printIntegerArray(float *, int); 
void barrier_sync(barrier_t *);
extern int compute_gold (float *, unsigned int);
Matrix allocate_matrix (int num_rows, int num_columns, int init);
void gauss_eliminate_using_pthreads (float *, float *, unsigned int);
int perform_simple_check (const Matrix);
void print_matrix (const Matrix);
float get_random_number (int, int);
int check_results (float *, float *, unsigned int, float);
void * gauss_eliminate(void *);
void printMatrix(const Matrix);
void printMatrix2(float *);


int
main (int argc, char **argv)
{
  /* Check command line arguments. */
  if (argc > 1)
    {
      printf ("Error. This program accepts no arguments. \n");
      exit (0);
    }

	//create a barrier object with a ount of NUM_THREADS
	//pthread_barrier_init(&barrier, NULL, NUM_THREADS);
	//pthread_barrier_init(&barrier2, NULL, NUM_THREADS);
  /* Matrices for the program. */
  Matrix A;			// The input matrix
  Matrix U_reference;		// The upper triangular matrix computed by the reference code
  Matrix U_mt;			// The upper triangular matrix computed by the pthread code

  /* Initialize the random number generator with a seed value. */
  srand (time (NULL));

  /* Allocate memory and initialize the matrices. */
  A = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);	// Allocate and populate a random square matrix
  U_reference = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	// Allocate space for the reference result
  U_mt = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	// Allocate space for the multi-threaded result

  /* Copy the contents of the A matrix into the U matrices. */
  for (int i = 0; i < A.num_rows; i++)
    {
      for (int j = 0; j < A.num_rows; j++)
	{
	  U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
	  U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
	}
    }

  printf ("Performing gaussian elimination using the reference code. \n");
  struct timeval start, stop;
  gettimeofday (&start, NULL);
  int status = compute_gold (U_reference.elements, A.num_rows);
  gettimeofday (&stop, NULL);
  float CPUruntime = (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000);
  printf ("CPU run time = %0.5f s. \n", CPUruntime);

  if (status == 0)
    {
      printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
      exit (0);
    }
  status = perform_simple_check (U_reference);	// Check that the principal diagonal elements are 1 
  if (status == 0)
    {
      printf ("The upper triangular matrix is incorrect. Exiting. \n");
      exit (0);
    }
  printf ("Single-threaded Gaussian elimination was successful. \n");
  int num_elements = A.num_rows;
  /*
  for (int q = 0; q < num_elements; q++) {
  	printIntegerArray(&U_reference.elements[q], num_elements);
  	printf("\n");
  }*/
  //printMatrix(U_reference);

  /* Perform the Gaussian elimination using pthreads. The resulting upper triangular matrix should be returned in U_mt */
  printf("Performing gaussian elimination using the multi-threaded code\n");
  gettimeofday(&start, NULL);
  gauss_eliminate_using_pthreads (U_mt.elements, A.elements, A.num_rows);
	gettimeofday(&stop, NULL);
	float parallelRunTime = (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000);
  /* check if the pthread result is equivalent to the expected solution within a specified tolerance. */
  int size = MATRIX_SIZE * MATRIX_SIZE;
  int res = check_results (U_reference.elements, U_mt.elements, size, 0.001f);
  printf ("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
  
  printf("Parallel run-time = %.5f\n", parallelRunTime);
  printf("Speedup = %.5f\n", CPUruntime/parallelRunTime);
  	
  	/*
  	printf("Original matrix: \n");
  	printMatrix(A);
  	printf("\n\n");
  	printf("Correct solution: \n");
  	printMatrix(U_reference);
  	printf("\n\n");
  	printf("PThreads solution: \n");
  	printMatrix(U_mt);
  	//printMatrix2(U_mt.elements);*/
  	

  /* Free memory allocated for the matrices. *
  free (A.elements);
  free (U_reference.elements);
  free (U_mt.elements);
  */

  return 0;
}


/* Write code to perform gaussian elimination using pthreads. */
void
gauss_eliminate_using_pthreads (float *U, float * A, unsigned int num_elements)
{
	unsigned int i, j, k; // loop counters
	
	///COPY CONTENTS OF MATRIX A INTO MATRIX U
	for (i = 0; i < num_elements; i ++) 
		for (j=0; j< num_elements; j ++) 
			U[num_elements * i + j] = A[num_elements*i + j];
	///
	
	pthread_t thread_id[NUM_THREADS]; //data structure to store thread ID
	pthread_attr_t attributes; //Thread attributes
	pthread_attr_init(&attributes); //initialize thread attributes to default
	
	ARGS_FOR_THREAD * args_for_thread[NUM_THREADS];
	
	for(i = 0; i < NUM_THREADS; i++) {
		args_for_thread[i] = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
		args_for_thread[i]->thread_id = i; //thread ID
		args_for_thread[i]-> num_elements = num_elements;
		args_for_thread[i]->matrixA = A;//can g oaway
		args_for_thread[i]->matrixU = U;
	}
	
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_create(&thread_id[i], &attributes, gauss_eliminate, (void*) args_for_thread[i]);
	}
	

	//DO THE JOIN HERE
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread_id[i], NULL);
	}
	
	
	//Free memory used by the threads
	/*
	for (i = 0; i < NUM_THREADS; i++) 
		free((void*)args_for_thread[i]);*/
	
}


//this function is executed by each thread individually
void * gauss_eliminate(void * args) {
	//type cast the thread's arguments
	ARGS_FOR_THREAD * my_args = (ARGS_FOR_THREAD *)args;
	int k, chunk, i, j;
	int num_elements = my_args->num_elements;
	//algorithm's outermost loop
	for (k = 0; k < num_elements; k++) {
		//printf("Iteration # %d \n", k+1);
		chunk = (int)floor((float)(num_elements - k) / (float) NUM_THREADS); //recalculate chunk size for each iteration

		//find this thread's individual offset for division
		my_args->offset = my_args->thread_id * chunk; 
		//do division
		
		if (my_args->thread_id < (NUM_THREADS - 1)) {
		
			for (j = (k + 1 + my_args->offset); j < (my_args->offset + k + 1 + chunk); j++) { 
				my_args->matrixU[num_elements*k + j] = (float)((float)my_args->matrixU[num_elements*k + j] / (float)(my_args->matrixU[num_elements*k + k]));
		
			}
	} else { //this takes care of the number of elements that the final thread must process
			for (j = (k + 1 + my_args->offset); j < num_elements; j++) { 
			my_args->matrixU[num_elements*k + j] = (float)((float)my_args->matrixU[num_elements*k + j] / (float)(my_args->matrixU[num_elements*k + k]));
			} //end for

		} //end else

		barrier_sync(&barrier);
	
		//do elimination
		if (my_args->thread_id < (NUM_THREADS - 1)) {
			for (i = (k + 1) + my_args->offset; i < (k + 1 + my_args->offset + chunk); i++) {
			
				for (j = k+1; j  < num_elements; j++) {
					my_args->matrixU[num_elements*i + j] = my_args->matrixU[num_elements*i+j] - (float)(my_args->matrixU[num_elements*i+k] * (float)my_args->matrixU[num_elements*k+j]);
				}

			my_args->matrixU[num_elements * i + k] = 0;

}
		} else {
			
			for (i = (k+1) + my_args->offset; i < num_elements; i++) {
				for (j = k+1; j < num_elements; j++) {
					my_args->matrixU[num_elements*i + j] = my_args->matrixU[num_elements*i+j] - ((float)my_args->matrixU[num_elements*i+k] * (float)my_args->matrixU[num_elements*k+j]);
					}
					
			my_args->matrixU[num_elements * i + k] = 0;
			}
		} //end else
	
	my_args->matrixU[num_elements*k + k] = 1;
	barrier_sync(&barrier2);
	
	
	} //outermost for loop with k

		pthread_exit((void *) 0);
} //end thread function


/* Function checks if the results generated by the single threaded and multi threaded versions match. */
int
check_results (float *A, float *B, unsigned int size, float tolerance)
{
  for (int i = 0; i < size; i++)
    if (fabsf (A[i] - B[i]) > tolerance)
      return 0;
  return 1;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix
allocate_matrix (int num_rows, int num_columns, int init)
{
  Matrix M;
  M.num_columns = M.pitch = num_columns;
  M.num_rows = num_rows;
  int size = M.num_rows * M.num_columns;

  M.elements = (float *) malloc (size * sizeof (float));
  for (unsigned int i = 0; i < size; i++)
    {
      if (init == 0)
	M.elements[i] = 0;
      else
	M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
    }
  return M;
}


/* Returns a random floating-point number between the specified min and max values. */ 
float
get_random_number (int min, int max)
{
  return (float)
    floor ((double)
	   (min + (max - min + 1) * ((float) rand () / (float) RAND_MAX)));
}

/* Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1. */
int
perform_simple_check (const Matrix M)
{
  for (unsigned int i = 0; i < M.num_rows; i++)
    if ((fabs (M.elements[M.num_rows * i + i] - 1.0)) > 0.001)
      return 0;
  return 1;
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

void printIntegerArray(float * array, int num_elements) {
	int i;
	printf("\n\n");
	for (i = 0; i < num_elements; i++) {
		printf("  %.2f  ", array[i]);
	}
	printf("\n\n");
}

void printMatrix(const Matrix M) {
	for (unsigned int i = 0; i < M.num_rows; i++) {
		for (unsigned int j = 0; j < M.num_rows; j++) {
			printf("%.3f ", M.elements[M.num_rows*i + j]);
		}
		printf("\n");
	}
}

void printMatrix2(float *U) {
	for (unsigned int i = 0; i < 15; i++) {
		for (unsigned int j = 0; j < 15; j++) {
			printf("%.3f ", U[15*i + j]);
		}
		printf("\n");
	}
}
