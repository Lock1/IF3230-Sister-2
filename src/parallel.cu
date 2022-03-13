#include <iostream>

int main(int argc, char const *argv[]) {
    std::cout << "abc\n";
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
