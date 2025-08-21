#include <iostream>
#include <cuda_runtime.h>
int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    std::cout << "CUDA Devices: " << count << std::endl;
    for(int i=0;i<count;i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name 
                  << " with " << prop.totalGlobalMem/1024/1024 << " MB" << std::endl;
    }
    return 0;
}
