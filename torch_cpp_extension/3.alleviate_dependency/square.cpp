#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <torch/csrc/utils/pybind.h>  // 内部会包含 pybind11/pybind11.h

// 声明 CUDA 核函数
void square_cuda_forward(void* input, void* output, int size);

// 对外暴露的接口函数
at::Tensor square_forward(const at::Tensor& input) {
    // 创建一个和 input 同形状、同 dtype 的张量当作输出
    auto output = at::zeros_like(input);
    
    // 调用 CUDA 核心函数
    square_cuda_forward(input.data_ptr(), output.data_ptr(), input.numel());
    
    return output;
}

// 使用 PyBind11 导出给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_forward", &square_forward, "Square forward (CUDA)");
}