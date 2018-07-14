/*  Purpose: Calculate definite integral using trapezoidal rule.
 *		Compute_gold() written by Dr. Kandasamy
 *		Parallel code written by Greg Matthews and Mark Klobukov
 *		2/13/2017
 *		Input:   a, b, n
 *		Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 *		Compile: gcc -o trap trap.c -lpthread -lm -std=c99
 *		Usage:   ./trap
 *
 *		Note:    The function f(x) is hardwired.
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define LEFT_ENDPOINT 5
#define RIGHT_ENDPOINT 1000
#define NUM_TRAPEZOIDS 100000000
#define NUM_THREADS 8

//data struture for passing arguments for each worker thread
typedef struct args_for_thread_s {
	float a; //beginning of integration
	float b; //end of integration
	float h; //trapezoid height;
	int thread_id; //thread id
	double my_sum; //partial sum of the given thread
	unsigned int start_index; //where on the x-axis thread starts
	unsigned int end_index; //where on the x-axis thread ends
} ARGS_FOR_THREAD;


double compute_using_pthreads(float, float, int, float);
double compute_gold(float, float, int, float);
void * compute_my_traps(void * );

int main(void) 
{
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);

	struct timeval start, stop;
	gettimeofday (&start, NULL);
	double reference = compute_gold(a, b, n, h);
	gettimeofday (&stop, NULL);
   printf("Reference solution computed on the CPU = %f \n", reference);
   float CPUruntime = (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000);
  printf ("CPU run time = %0.5f s. \n", CPUruntime);

	/* Write this function to complete the trapezoidal on the GPU. */
	gettimeofday(&start, NULL);
	double pthread_result = compute_using_pthreads(a, b, n, h);
	gettimeofday(&stop, NULL);
	printf("Solution computed using pthreads = %f \n", pthread_result);
	float parallelRunTime = (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000);
	printf("Parallel run-time = %.5f\n", parallelRunTime);
  printf("Speedup = %.5f\n", CPUruntime/parallelRunTime);
		
} 


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 * Output: (x+1)/sqrt(x*x + x + 1)

 */
float f(float x) {
		  return (x + 1)/sqrt(x*x + x + 1);
}  /* f */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double compute_gold(float a, float b, int n, float h) {
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;
   for (k = 1; k <= n-1; k++) {
     integral += f(a+k*h);
   }
   integral = integral*h;

   return integral;
}  

/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_using_pthreads(float a, float b, int n, float h)
{
			double sum = 0.0;		
			pthread_t thread_id[NUM_THREADS];
			pthread_attr_t attributes; //thread attr
			pthread_attr_init(&attributes); //init thread attributes to default
			ARGS_FOR_THREAD * args_for_thread[NUM_THREADS];
			unsigned int i;
			for (i = 0; i < NUM_THREADS; i++) {
				args_for_thread[i] = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
				args_for_thread[i]->a = a;
				args_for_thread[i]->b = b;
				args_for_thread[i]->h = h;
				args_for_thread[i]->thread_id = i; //thread id
				args_for_thread[i]->my_sum = 0.0; //init sum
				args_for_thread[i]->start_index = 1 + (NUM_TRAPEZOIDS/NUM_THREADS)*i;
				args_for_thread[i]->end_index = 1 + (NUM_TRAPEZOIDS/NUM_THREADS) * (i+1);
				}
			
			//create threads
			for (i = 0; i< NUM_THREADS; i++) {
					pthread_create(&thread_id[i], &attributes, compute_my_traps, (void*) args_for_thread[i]);
			}
			
			
			//join threads
			for (i = 0; i < NUM_THREADS; i++) {
					pthread_join(thread_id[i], NULL);
			}
			
			for (i = 0; i < NUM_THREADS; i++) {
					sum += args_for_thread[i]->my_sum ;
			}
			sum = (sum + (f(a) + f(b)) /2.0) * h; 
			
			return sum;
}
			
void * compute_my_traps(void * args) {
		//types cast the thread's arguments
		ARGS_FOR_THREAD * my_args = (ARGS_FOR_THREAD*) args;
		float h = my_args->h;
		float a = my_args->a;
		unsigned int end = my_args->end_index;
		unsigned int start = my_args->start_index;
		//take care of a potentially smaller chunk at the end
		if (my_args->thread_id == (NUM_THREADS - 1)) {
			end = NUM_TRAPEZOIDS;
		}

		for (unsigned int k = start; k < end; k++) {
				my_args->my_sum += f(a  + k*h);
		}
		//multiply each thread's sum by h and add to the overall sum in the thread-creating function
		pthread_exit((void*) 0);
}
