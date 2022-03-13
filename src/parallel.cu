#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>

#include "serial_lib.hpp"
using namespace std;

__global__ void convolution(Matrix *d_kernel) {
    __shared__ Matrix kernel;
    kernel = *d_kernel;
    __syncthreads();

    // printf("<%d, %d> %d\n", threadIdx.x, threadIdx.y, kernel.mat[threadIdx.x][threadIdx.y]);
}

void compute_convolution(ifstream &fs) {
    int kernel_row, kernel_col, target_row, target_col, num_targets;
    Matrix kernel;

    fs >> kernel_row >> kernel_col;
    kernel = input_matrix(fs, kernel_row, kernel_col);

    fs >> num_targets >> target_row >> target_col;

    Matrix *d_kernel;
    cudaMalloc((void **) &d_kernel, sizeof(Matrix));
    cudaMemcpy(d_kernel, &kernel, sizeof(Matrix), cudaMemcpyHostToDevice);

    dim3 gridDim(1);
    dim3 blockDim(16, 16);
    convolution<<<gridDim, blockDim>>>(d_kernel);
    cudaDeviceSynchronize();
}


int main(int argc, char const *argv[]) {
    ifstream fs(argv[1]);

    if (argc > 1 && fs.is_open()) {
        compute_convolution(fs);
        fs.close();
    }
    else {
        cout << "parallel: Failed to open file\n";
        exit(1);
    }

    return 0;
}




// __global__ void add(int *d_a, int *d_b, int *d_c) {
//   *d_c = *d_a + *d_b;
// }

// int main() {
//   int a, b, c; // host copies of variables a, b & c
//   int *d_a, *d_b, *d_c; // device copies of variables a, b & c
//
//   int size = sizeof(int); // Allocate space for device copies of a, b, c
//   cudaMalloc((void **)&d_a, size);
//   cudaMalloc((void **)&d_b, size);
//   cudaMalloc((void **)&d_c, size);
//
//   // Setup input values
//   c = 0;
//   a = 10;
//   b = 11;
//
//   // Copy inputs to device
//   cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
//
//   // Launch add() kernel on GPU
//   add<<<1,1>>>(d_a, d_b, d_c);
//
//   // Copy result back to host
//   cudaError err = cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
//   if(err!=cudaSuccess) {
//     printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
//   }
//
//   printf("result is %d\n", c);
//
//   if (c != 21) {
//     printf("Something wrong.\n");
//   }
//
//   // Cleanup
//   cudaFree(d_a);
//   cudaFree(d_b);
//   cudaFree(d_c);
//   return 0;
// }
