#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

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

void square_cuda_forward(void *input, void *output, int size);

// 需要暴露给 Python 的接口函数
torch::Tensor square_forward(torch::Tensor input)
{
    // 创建一个和 input 同形状、同 dtype 的张量当作输出
    auto output = torch::zeros_like(input);

    // 调用 CUDA 核心函数
    square_cuda_forward(input.data_ptr(), output.data_ptr(), input.numel());

    return output;
}

// 使用 PyBind11 导出给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("square_forward", &square_forward, "Square forward (CUDA)");
}
