
#include <stdio.h>
#include <math.h>
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" void compute_scan_gold( float* reference, float* idata, const unsigned int len);
extern "C" int compact_stream_gold(float *reference, float *idata, unsigned int len);

// Reference code for exclusive scan
void compute_scan_gold( float* reference, float* idata, const unsigned int len) 
{
  reference[0] = 0;
  double total_sum = 0;
  unsigned int i;
  for( i = 1; i < len; ++i) 
  {
      total_sum += idata[i-1];
      reference[i] = idata[i-1] + reference[i-1];
  }
  // Here it should be okay to use != because we have integer values
  // in a range where float can be exactly represented
  if (total_sum != reference[i-1])
      printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
  
}

// Compact the input stream by filtering out only positive values greater than zero
int compact_stream_gold(float *reference, float *idata, unsigned int len)
{
	int n = 0; // Number of elements in the compacted stream
	for(unsigned int i = 0; i < len; i++){
		if(idata[i] > 0.0){
			reference[n++] = idata[i];
		}
	}
	
    return n; // Return the number of elements in the compacted stream
}
