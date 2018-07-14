#include <stdlib.h>


extern "C"
void computeGold( float*, const float*, const float*, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! P = N convolved with M
//! @param P          reference solution
//! @param M          matrix M
//! @param N          matrix N
//! @param kernel_size         height and width of matrix M which is fixed at 5
//! @param hN         height of matrices N and P
//! @param wN         width of matrices N and P
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* P, const float* M, const float* N, unsigned int hN, unsigned int wN)
{
	// For each element in the result matrix matrix
	for (unsigned int i = 0; i < hN; ++i){
        for (unsigned int j = 0; j < wN; ++j) {
			double sum = 0;
			// check the start and end values of m and n to prevent overrunning the 
			//  matrix edges
			unsigned int mbegin = (i < 2)? 2 - i : 0;
			unsigned int mend = (i > (hN - 3))?
									hN - i + 2 : 5;
			unsigned int nbegin = (j < 2)? 2 - j : 0;
			unsigned int nend = (j > (wN - 3))?
									(wN-j) + 2 : 5;
			// overlay M over N centered at element (i,j).  For each 
			//  overlapping element, multiply the two and accumulate
			for(unsigned int m = mbegin; m < mend; ++m) {
				for(unsigned int n = nbegin; n < nend; n++) {
					sum += M[m * 5 + n] * 
							N[wN*(i + m - 2) + (j+n - 2)];
				}
			}
			// store the result
			P[i*wN + j] = (float)sum;
        }
	}
}

