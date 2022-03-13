#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>

#define BLOCK_LENGTH 16

#include "serial_lib.hpp"
using namespace std;

__device__ int supression_op(Matrix &kernel, Matrix &target, int row, int col) {
	int intermediate_sum = 0;
	for (int i = 0; i < kernel.row_eff; i++)
		for (int j = 0; j < kernel.col_eff; j++)
			intermediate_sum += kernel.mat[i][j] * target.mat[row + i][col + j];

	return intermediate_sum;
}

__global__ void convolution(Matrix *d_kernel, Matrix *d_target, Matrix *d_result) {
    __shared__ Matrix kernel;
    kernel = *d_kernel;
    int target_row = d_target[0].row_eff;
    int target_col = d_target[0].col_eff;
    __syncthreads();

    for (int i = threadIdx.x; i < target_row; i += BLOCK_LENGTH) {
        for (int j = threadIdx.y; j < target_col; j += BLOCK_LENGTH) {
            int inter_sum = supression_op(kernel, d_target[blockIdx.x], i, j);
            d_result[blockIdx.x].mat[i][j] = inter_sum;
        }
    }
}

int cmpfunc(void const *a, void const *b) {
   return *(int*)a - *(int*)b;
}

void compute_convolution(ifstream &fs) {
    int kernel_row, kernel_col, target_row, target_col, num_targets;
    Matrix kernel;

    // File stream processing
    fs >> kernel_row >> kernel_col;
    kernel = input_matrix(fs, kernel_row, kernel_col);
    fs >> num_targets >> target_row >> target_col;

    Matrix *target_container = new Matrix[num_targets];
    for (int i = 0; i < num_targets; i++)
        target_container[i] = input_matrix(fs, target_row, target_col);

    Matrix *result_container = new Matrix[num_targets];
    int res_row = target_row - kernel_row + 1;
    int res_col = target_col - kernel_col + 1;
    for (int i = 0; i < num_targets; i++)
        init_matrix(result_container[i], res_row, res_col);

    // Host to device memory copy
    Matrix *d_kernel;
    cudaMalloc((void **) &d_kernel, sizeof(Matrix));
    cudaMemcpy(d_kernel, &kernel, sizeof(Matrix), cudaMemcpyHostToDevice);

    Matrix *d_target;
    cudaMalloc((void **) &d_target, sizeof(Matrix)*num_targets);
    cudaMemcpy(d_target, target_container, sizeof(Matrix)*num_targets, cudaMemcpyHostToDevice);

    Matrix *d_result;
    cudaMalloc((void **) &d_result, sizeof(Matrix)*num_targets);
    cudaMemcpy(d_result, result_container, sizeof(Matrix)*num_targets, cudaMemcpyHostToDevice);

    // Device execution
    dim3 gridDim(num_targets);
    dim3 blockDim(BLOCK_LENGTH, BLOCK_LENGTH);
    convolution<<<gridDim, blockDim>>>(d_kernel, d_target, d_result);
    cudaDeviceSynchronize();

    // Result processing
    cudaError err = cudaMemcpy(result_container, d_result, sizeof(Matrix)*num_targets, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    int *matrix_ranges = new int[num_targets];
    for (int i = 0; i < num_targets; i++)
        matrix_ranges[i] = get_matrix_datarange(result_container[i]);

    qsort(matrix_ranges, num_targets, sizeof(int), cmpfunc);

    int median       = get_median(matrix_ranges, num_targets);
    int floored_mean = get_floored_mean(matrix_ranges, num_targets);

    printf("%d\n%d\n%d\n%d\n",
            matrix_ranges[0],
            matrix_ranges[num_targets - 1],
            median,
            floored_mean);


    // Release memory
    delete target_container;
    delete result_container;
    cudaFree(d_kernel);
    cudaFree(d_target);
    cudaFree(d_result);
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
