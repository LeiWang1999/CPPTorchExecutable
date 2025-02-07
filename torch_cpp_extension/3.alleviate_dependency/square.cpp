#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <torch/csrc/utils/pybind.h>

void square_cuda_forward(void* input, void* output, int size);

at::Tensor square_forward(const at::Tensor& input) {
    auto output = at::zeros_like(input);

    square_cuda_forward(input.data_ptr(), output.data_ptr(), input.numel());
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_forward", &square_forward, "Square forward (CUDA)");
}
