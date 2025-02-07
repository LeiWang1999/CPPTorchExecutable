import torch
from torch.utils.cpp_extension import load
import time
import pathlib

dir_path = pathlib.Path(__file__).parent.absolute()
print(f"dir_path: {dir_path}")

build_dir = f"{dir_path}/build"

if not pathlib.Path(build_dir).exists():
    pathlib.Path(build_dir).mkdir(parents=True)
else:
    # Clean the build directory
    for file in pathlib.Path(build_dir).glob("*"):
        file.unlink()

start = time.time()

square_cuda = load(
    name="square_cuda",
    sources=[f"{dir_path}/square_kernel.cu"],
    verbose=False,
    build_directory=build_dir
)

end = time.time()
print(f"Time taken: {end - start} seconds")

x = torch.randn((5,), device="cuda")
y = square_cuda.square_forward(x)
print("Input:", x)
print("Output:", y)
