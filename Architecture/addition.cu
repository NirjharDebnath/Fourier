#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    int n = 1 << 16;
    float alpha = 2.0f;

    std::vector<float> h_x(n), h_y(n);
    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(2 * i);
    }

    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform y = alpha*x + y
    cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);

    // Copy back result
    cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_y[i] != 2.0f * i + alpha * i) {
            std::cerr << "Mismatch at " << i << ": " << h_y[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) std::cout << "AXPY successful!" << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
