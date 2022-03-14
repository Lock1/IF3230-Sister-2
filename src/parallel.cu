#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>

#define BLOCK_LENGTH 16

#include "serial_lib.hpp"
using namespace std;

// Debugging function
__global__ void device_print_array(int *arr) {
    printf("<%d> %d\n", threadIdx.x, arr[threadIdx.x]);
}

__device__ void internal_print_array(int *n, int left, int right) {
    // WARNING : SPECIFIC DEBUGGING PURPOSE ONLY,
    //  printf order CAN be semi-randomized in multiple thread execution
    for (int i = left; i <= right; i++)
        printf("%d ", n[i]);
    printf("\n");
}

__device__ void sort_sanity_check(int *n, int left, int right) {
    for (int i = left; i < right; i++) {
        if (n[i] > n[i+1]) {
            printf("<%d> sanity check error: index %d, (%d | %d)\n", threadIdx.x, i, n[i], n[i+1]);
        }
    }
}


// Range calculation
__device__ int range_calculation(Matrix *d_result) {
    int max = DATAMIN;
	int min = DATAMAX;
	for (int i = 0; i < d_result[blockIdx.x].row_eff; i++) {
		for (int j = 0; j < d_result[blockIdx.x].col_eff; j++) {
			int el = d_result[blockIdx.x].mat[i][j];
			if (el > max) max = el;
			if (el < min) min = el;
		}
	}

	return max - min;
}


// Convolution
__device__ int supression_op(Matrix &kernel, Matrix &target, int row, int col) {
	int intermediate_sum = 0;
	for (int i = 0; i < kernel.row_eff; i++)
		for (int j = 0; j < kernel.col_eff; j++)
			intermediate_sum += kernel.mat[i][j] * target.mat[row + i][col + j];

	return intermediate_sum;
}

__global__ void convolution(Matrix *d_kernel, Matrix *d_target, Matrix *d_result, int *d_ranges) {
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

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        d_ranges[blockIdx.x] = range_calculation(d_result);
}


// Sorting
__device__ void merge_array(int *n, int left, int mid, int right) {
    if (left < right) {
    	int n_left    = mid - left + 1;
    	int n_right   = right - mid;
    	int iter_left = 0, iter_right = 0, iter_merged = left;

        int *arr_left  = new int[n_left];
        int *arr_right = new int[n_right];

        // Data copying
    	for (int i = 0; i < n_left; i++) {
    		arr_left[i] = n[i + left];
    	}

    	for (int i = 0; i < n_right; i++) {
    		arr_right[i] = n[i + mid + 1];
    	}

        // Merge sort insertion
    	while (iter_left < n_left && iter_right < n_right) {
    		if (arr_left[iter_left] <= arr_right[iter_right]) {
    			n[iter_merged] = arr_left[iter_left++];
    		} else {
    			n[iter_merged] = arr_right[iter_right++];
    		}
    		iter_merged++;
    	}

        // Leftover merging
    	while (iter_left < n_left)  {
    		n[iter_merged++] = arr_left[iter_left++];
    	}
    	while (iter_right < n_right) {
    		n[iter_merged++] = arr_right[iter_right++];
    	}

        delete arr_left;
        delete arr_right;
    }
}

__device__ int get_chunk_size(int size, int tid) {
    /* Get chunk size at specified size and threadIdx value
       Reduce chunk size if threadIdx more than available "extra" partition count
       Ex. Thread count : 5, Array size : 8
       Notation, t[i] : n, t[i] thread and threadIdx, n partition / chunk size
       t[0] : 2, t[1] : 2, t[2] : 2, t[3] : 1, t[4] : 1
    */
    int chunk_size               = size / blockDim.x + 1;         // Special "ceil", c(1.0) = 2
    int no_extra_partition_count = blockDim.x*chunk_size - size;

    if (tid > blockDim.x - no_extra_partition_count - 1)
        chunk_size--;

    return chunk_size;
}

__device__ int get_left_index(int size, int tid) {
    int chunk_size               = size / blockDim.x + 1;
    int no_extra_partition_count = blockDim.x*chunk_size - size;
    int split_index              = blockDim.x - no_extra_partition_count - 1;

    int index;
    // index = chunk_size*split_index + (tid - split_index) * (chunk_size - 1) + 1;
    // index = chunk_size*split_index + tid * chunk_size - tid - split_index * chunk_size + split_index + 1;
    if (tid > split_index)
        index = tid * (chunk_size - 1) + split_index + 1;
    else
        index = tid * chunk_size;

    return index;
}

__device__ void bubble_sort_helper(int *n, int left, int right) {
    if (left < right) {
        char pass_success = 0;

        while (1) {
            pass_success = 1;

            for (int i = left; i < right; i++) {
                for (int j = i + 1; j <= right; j++) {
                    if (n[i] > n[j]) {
                        int temp     = n[i];
                        n[i]         = n[j];
                        n[j]         = temp;
                        pass_success = 0;
                    }
                }
            }

            if (pass_success)
                break;
        }
    }
}

__global__ void merge_sort_device(int *n, int *csize_arr, int size) {
    int current_chunk_size = get_chunk_size(size, threadIdx.x);

    // Modified merge sort from MPI
    int pow             = 0;
    int current_stride  = (1 << pow);
    int reciever_stride = (1 << (pow + 1));
    int left            = get_left_index(size, threadIdx.x);

    // Initial pass, using bubble sort
    int right           = left + current_chunk_size - 1;
    bubble_sort_helper(n, left, right);
    __syncthreads();

    // Merge sort
    while (current_stride <= size) {
        if (threadIdx.x % reciever_stride == 0) {
            int rank_target = threadIdx.x + current_stride;
            if (rank_target < BLOCK_LENGTH*BLOCK_LENGTH) {
                int mid                = left + current_chunk_size - 1; // Old right -> mid
                int cont_recv_size     = csize_arr[rank_target];
                current_chunk_size     += cont_recv_size;
                csize_arr[threadIdx.x] = current_chunk_size;

                int right              = left + current_chunk_size - 1; // New right
                merge_array(n, left, mid, right);
            }
        }

        // Moving to next layer, need to wait all thread sorting process done
        __syncthreads();
        pow++;
        current_stride  = (1 << pow);
        reciever_stride = (1 << (pow + 1));
    }
}

__global__ void init_chunk_size_array(int *chunk_array, int target_sort_array_size) {
    chunk_array[threadIdx.x] = get_chunk_size(target_sort_array_size, threadIdx.x);
}

void merge_sort(int *d_ptr, int size) {
    int *d_thread_chunk_sizes;
    cudaMalloc((void **) &d_thread_chunk_sizes, sizeof(int)*BLOCK_LENGTH*BLOCK_LENGTH);
    init_chunk_size_array<<<1, BLOCK_LENGTH*BLOCK_LENGTH>>>(d_thread_chunk_sizes, size);
    cudaDeviceSynchronize();

    merge_sort_device<<<1, BLOCK_LENGTH*BLOCK_LENGTH>>>(d_ptr, d_thread_chunk_sizes, size);
    cudaFree(d_thread_chunk_sizes);
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

    int *d_ranges;
    cudaMalloc((void **) &d_ranges, sizeof(int)*num_targets);

    // Device execution, convolution & range calculation
    dim3 gridDim(num_targets);
    dim3 blockDim(BLOCK_LENGTH, BLOCK_LENGTH);
    convolution<<<gridDim, blockDim>>>(d_kernel, d_target, d_result, d_ranges);
    cudaDeviceSynchronize();

    // Device execution, sorting
    merge_sort(d_ranges, num_targets);
    cudaDeviceSynchronize();

    // Result processing
    int *matrix_ranges = new int[num_targets];
    cudaError err = cudaMemcpy(matrix_ranges, d_ranges, sizeof(int)*num_targets, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

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
    delete matrix_ranges;
    cudaFree(d_kernel);
    cudaFree(d_target);
    cudaFree(d_result);
    cudaFree(d_ranges);
}


int main(int argc, char const *argv[]) {
    clock_t timer;
    timer = clock();
    ifstream fs(argv[1]);

    if (argc > 1 && fs.is_open()) {
        compute_convolution(fs);
        fs.close();
    }
    else {
        cout << "parallel: Failed to open file\n";
        exit(1);
    }

    if (argc > 2) {
        timer = clock() - timer;
        double time_elapsed = ((double) timer) / CLOCKS_PER_SEC;
        printf("Time elapsed %f\n", time_elapsed);
    }

    return 0;
}
