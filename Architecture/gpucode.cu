#include <iostream>
#include <stdio.h>

// Macro for checking CUDA API calls for errors
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    std::cout << "Hello World from CPU!" << std::endl;

    // Launch the kernel
    helloFromGPU<<<2, 10>>>();

    // Check for any errors that might have occurred during kernel launch
    // This is the most important check!
    CUDA_CHECK(cudaGetLastError());

    // Reset the device and check for errors
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}