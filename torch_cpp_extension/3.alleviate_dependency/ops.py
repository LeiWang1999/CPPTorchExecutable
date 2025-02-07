import torch
from torch.utils.cpp_extension import load
import os
import time

start = time.time()

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
# os.environ["TORCH_INDUCTOR_CPP_WRAPPER"] = "True"

square_cuda = load(
    name="square_cuda",                      # 编译后的扩展模块名
    sources=["square.cpp", "square_kernel.cu"],  # 要编译的源文件
    verbose=False,   # 打印编译输出，便于调试
    # extra_cflags=["-DTORCH_INDUCTOR_CPP_WRAPPER"],  # 传递额外的编译参数
    # extra_cuda_cflags=["-DTORCH_INDUCTOR_CPP_WRAPPER"]  # 传递额外的编译参数
)

end = time.time()
print(f"Time taken: {end - start} seconds")

# 现在就可以使用 square_cuda 里的绑定函数
x = torch.randn((5,), device="cuda")
y = square_cuda.square_forward(x)
print("Input:", x)
print("Output:", y)