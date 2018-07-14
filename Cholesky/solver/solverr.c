/* 
Code for the equation solver. 
Author: Naga Kandasamy 
Date: 5/7/2013

Compile as follows:

gcc -o solver solver.c solver_gold.c -fopenmp -std=c99 -lm -lpthread
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "grid.h" // This file defines the grid data structure

#define NUM_THREADS 4
extern int compute_gold(GRID_STRUCT *);
int compute_using_pthread_jacobi(GRID_STRUCT *);
int compute_using_pthread_red_black(GRID_STRUCT *);

/* This function prints the grid on the screen */
void display_grid(GRID_STRUCT *my_grid)
{
	for(int i = 0; i < my_grid->dimension; i++)
		for(int j = 0; j < my_grid->dimension; j++)
			printf("%f \t", my_grid->element[i * my_grid->dimension + j]);
   		
		printf("\n");
}


// This function prints out statistics for the converged values, including min, max, and average. */
void print_statistics(GRID_STRUCT *my_grid)
{
		// Print statistics for the CPU grid
		float min = INFINITY;
		float max = 0.0;
		double sum = 0.0; 
		for(int i = 0; i < my_grid->dimension; i++){
			for(int j = 0; j < my_grid->dimension; j++){
				sum += my_grid->element[i * my_grid->dimension + j]; // Compute the sum
				if(my_grid->element[i * my_grid->dimension + j] > max) max = my_grid->element[i * my_grid->dimension + j]; // Determine max
				if(my_grid->element[i * my_grid->dimension + j] < min) min = my_grid->element[i * my_grid->dimension + j]; // Determine min
				 
			}
		}

	printf("AVG: %f \n", sum/(float)my_grid->num_elements);
	printf("MIN: %f \n", min);
	printf("MAX: %f \n", max);

	printf("\n");
}

/* This function creates a grid of random floating point values bounded by UPPER_BOUND_ON_GRID_VALUE */
void create_grids(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2, GRID_STRUCT *grid_3)
{
	printf("Creating a grid of dimension %d x %d. \n", grid_1->dimension, grid_1->dimension);
	grid_1->element = (float *)malloc(sizeof(float) * grid_1->num_elements);
	grid_2->element = (float *)malloc(sizeof(float) * grid_2->num_elements);
	grid_3->element = (float *)malloc(sizeof(float) * grid_3->num_elements);

	srand((unsigned)time(NULL)); // Seed the the random number generator 
	
	float val;
	for(int i = 0; i < grid_1->dimension; i++)
		for(int j = 0; j < grid_1->dimension; j++){
			val =  ((float)rand()/(float)RAND_MAX) * UPPER_BOUND_ON_GRID_VALUE; // Obtain a random value
			grid_1->element[i * grid_1->dimension + j] = val; 	
			grid_2->element[i * grid_2->dimension + j] = val; 
			grid_3->element[i * grid_3->dimension + j] = val; 
			
		}
}

/* Edit this function to use the jacobi method of solving the equation. The final result should be placed in the final_grid_1 data structure */
int compute_using_pthread_jacobi(GRID_STRUCT *grid_2)
{
	int num_iter = 0;
	int done = 0;
	float diff;
	float temp; 
	
	while(!done){ // While we have not converged yet 
		printf("Iteration %d; ", num_iter);
		diff = 0;	
		for(int i = 1; i < (my_grid->dimension-1); i++){
			for(int j = 1; j < (my_grid->dimension-1); j++){
				temp = my_grid->element[i * my_grid->dimension + j];
			  	// Apply the update rule	
				my_grid->element[i * my_grid->dimension + j] = 0.20*(my_grid->element[i * my_grid->dimension + j] + 
																					 my_grid->element[(i - 1) * my_grid->dimension + j] + 
																					 my_grid->element[(i + 1) * my_grid->dimension + j] + 
							  														 my_grid->element[i * my_grid->dimension + (j + 1)] + 
																					 my_grid->element[i * my_grid->dimension + (j - 1)]);
				diff = diff + fabs(my_grid->element[i * my_grid->dimension + j] - temp);  
			}
		}
		/* End of an iteration. Check for convergence */
		num_iter++;
		printf("diff = %f \n", diff);
		if((float)diff/((float)(my_grid->dimension*my_grid->dimension)) < (float)TOLERANCE) done = 1;
	}
	return num_iter;
}

/* Edit this function to use the red-black method of solving the equation. The final result should be placed in the final_grid_2 data structure */
int compute_using_pthread_red_black(GRID_STRUCT *my_grid)
{
		int num_iter = 0, flag = 0;
	int i, j; 
	int done = 0;
	float diff;
	float temp; 
	
	while(!done){ // While we have not converged yet 
		printf("Iteration %d; ", num_iter);
		diff = 0;	
		
		for (int k = 0; k < 2; k++) {//k=0-->blacks, k=1-->reds
		
#pragma omp parallel for num_threads(NUM_THREADS) default(none) private(i, j, temp) shared(k, my_grid) reduction( + : diff)
			for(i = 1; i < (my_grid->dimension-1); i++){
				for(j = (i+k) % 2 + 1; j < (my_grid->dimension-1); j+=2){
					temp = my_grid->element[i * my_grid->dimension + j];
						// Apply the update rule	
					my_grid->element[i * my_grid->dimension + j] = 0.20*(my_grid->element[i * my_grid->dimension + j] + 
																						 my_grid->element[(i - 1) * my_grid->dimension + j] + 
																						 my_grid->element[(i + 1) * my_grid->dimension + j] + 
																							 my_grid->element[i * my_grid->dimension + (j + 1)] + 
																						 my_grid->element[i * my_grid->dimension + (j - 1)]);
					diff = diff + fabs(my_grid->element[i * my_grid->dimension + j] - temp);  
				}//end j
			}//end i
	}//end k
		/* End of an iteration. Check for convergence */
		num_iter++;
		printf("diff = %f \n", diff);
		if((float)diff/((float)(my_grid->dimension*my_grid->dimension)) < (float)TOLERANCE) done = 1;
	}
	return num_iter;			
}

		
/* The main function */
int main(int argc, char **argv)
{	
	/* Generate the grids and populate them with the same set of random values. */
	GRID_STRUCT *grid_1 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 
	GRID_STRUCT *grid_2 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 
	GRID_STRUCT *grid_3 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 

	grid_1->dimension = GRID_DIMENSION;
	grid_1->num_elements = grid_1->dimension * grid_1->dimension;
	grid_2->dimension = GRID_DIMENSION;
	grid_2->num_elements = grid_2->dimension * grid_2->dimension;
	grid_3->dimension = GRID_DIMENSION;
	grid_3->num_elements = grid_3->dimension * grid_3->dimension;


 	create_grids(grid_1, grid_2, grid_3);

	struct timeval start, stop; //for measuring time
	// Compute the reference solution
	printf("Using the single threaded version to solve the grid. \n");
	gettimeofday(&start, NULL);
	int num_iter = compute_gold(grid_1);	
	gettimeofday(&stop, NULL);
	printf("Convergence achieved after %d iterations. \n", num_iter);
	float serialTime = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("Serial program run time = %.4f s. \n", serialTime);
	
	// Use pthreads to solve the equation uisng the red-black parallelization technique
	printf("Using the OpenMP implementation to solve the grid using the red-black parallelization method. \n");
	gettimeofday(&start, NULL);
	num_iter = compute_using_pthread_red_black(grid_2);
	gettimeofday(&stop, NULL);
	printf("Convergence achieved after %d iterations. \n", num_iter);
	float redblackTime = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("Red-Black method run time = %.4f s. \n", redblackTime);
	printf("Red-Black Speedup = %.4f \n", serialTime/redblackTime);	
	
	// Use pthreads to solve the equation using the jacobi method in parallel
	printf("Using the OpenMP implementation to solve the grid using the jacobi method. \n");
	gettimeofday(&start, NULL);
	num_iter = compute_using_pthread_jacobi(grid_3);
	gettimeofday(&stop, NULL);
	printf("Convergence achieved after %d iterations. \n", num_iter);
	float jacobiTime = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("Jacobi method run time = %.4f s. \n", jacobiTime);
	printf("Jacobi Speedup = %.4f \n", serialTime/jacobiTime);	


	// Print key statistics for the converged values
	printf("\n");
	printf("Reference: \n");
	print_statistics(grid_1);

	printf("Red-black: \n");
	print_statistics(grid_2);
		
	printf("Jacobi: \n");
	print_statistics(grid_3);

	// Free the grid data structures
	free((void *)grid_1->element);	
	free((void *)grid_1); 
	
	free((void *)grid_2->element);	
	free((void *)grid_2);

	free((void *)grid_3->element);	
	free((void *)grid_3);

	exit(0);
}
