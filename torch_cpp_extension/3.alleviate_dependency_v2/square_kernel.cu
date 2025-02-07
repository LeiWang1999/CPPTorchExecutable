#include <cuda.h>
#include <cuda_runtime.h>

void square_cuda_forward(void* input, void* output, int size);

// 一个简单的平方kernel
__global__ void square_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    }
}

// 供 C++ 调用的函数
void square_cuda_forward(void* input, void* output, int size) {

    // 设置线程配置
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // 调用 kernel
    square_kernel<<<blocks, threads>>>(
        static_cast<const float*>(input),
        static_cast<float*>(output),
        size
    );
}
