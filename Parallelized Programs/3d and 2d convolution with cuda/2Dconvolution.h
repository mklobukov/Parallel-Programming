
#ifndef _2DCONVOLVE_H_
#define _2DCONVOLVE_H_

// Thread block size
#define THREAD_BLOCK_SIZE 32
#define KERNEL_SIZE 5
#define MATRIX_SIZE 2048

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

#endif

