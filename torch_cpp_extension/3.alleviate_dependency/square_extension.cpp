#include <torch/extension.h>

void square_cuda_forward(void* input, void* output, int size);

// 需要暴露给 Python 的接口函数
torch::Tensor square_forward(torch::Tensor input) {
    // 创建一个和 input 同形状、同 dtype 的张量当作输出
    auto output = torch::zeros_like(input);

    // 调用 CUDA 核心函数
    square_cuda_forward(input.data_ptr(), output.data_ptr(), input.numel());

    return output;
}

// 使用 PyBind11 导出给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_forward", &square_forward, "Square forward (CUDA)");
}
