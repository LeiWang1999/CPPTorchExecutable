#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

void square_cuda_forward(void* input, void* output, int size);

__global__ void square_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    }
}

void square_cuda_forward(void* input, void* output, int size) {

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    square_kernel<<<blocks, threads>>>(
        static_cast<const float*>(input),
        static_cast<float*>(output),
        size
    );
}

void square_cuda_forward(void *input, void *output, int size);

torch::Tensor square_forward(torch::Tensor input)
{
    auto output = torch::zeros_like(input);

    square_cuda_forward(input.data_ptr(), output.data_ptr(), input.numel());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("square_forward", &square_forward, "Square forward (CUDA)");
}
