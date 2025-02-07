from tvm.script import ir as I
from tvm.script import tir as T
import tvm
import torch
from torch.utils.dlpack import to_dlpack

@I.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer((1, 1024), "float16"),
        B: T.Buffer((1024, 512), "int8"),
        Scale: T.Buffer((1024, 1), "float16"),
        Zeros: T.Buffer((1024, 1), "float16"),
        D: T.Buffer((1, 1024), "float16"),
    ):
        T.func_attr(
            {
                "dequantize_info": {
                    "B_decode": {
                        "decode_block": "B_decode",
                        "fast_decoding": T.bool(False),
                        "group_size": 1024,
                        "source_format": {"bits": 4, "format": "uint"},
                        "storage_dtype": "int8",
                        "target_format": "float16",
                        "with_scaling": T.bool(True),
                        "with_zeros": T.bool(True),
                        "zeros_type": "original",
                    }
                },
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        B_decode_local = T.alloc_buffer((1024, 1024), "float16", scope="local")
        A_local = T.alloc_buffer((1, 1024), "float16", scope="local")
        B_local = T.alloc_buffer((1024, 512), "int8", scope="local")
        C_local = T.alloc_buffer((1, 1024), "float16", scope="local")
        for ax0_0 in T.thread_binding(512, thread="blockIdx.x"):
            for ax0_1 in T.thread_binding(2, thread="threadIdx.y"):
                for ax1_0 in range(2):
                    for ax1_1 in T.thread_binding(64, thread="threadIdx.x"):
                        for ax0 in range(1):
                            for ax1 in T.vectorized(4):
                                with T.block("B_local"):
                                    v0 = T.axis.spatial(1024, ax0_0 * 2 + ax0_1 + ax0)
                                    v1 = T.axis.spatial(
                                        512, ax1_0 * 256 + ax1_1 * 4 + ax1
                                    )
                                    T.reads(B[v0, v1])
                                    T.writes(B_local[v0, v1])
                                    B_local[v0, v1] = B[v0, v1]
                        for ax0, ax1 in T.grid(1, 8):
                            with T.block("B_decode_local"):
                                v0 = T.axis.spatial(1024, ax0_0 * 2 + ax0_1 + ax0)
                                v1 = T.axis.spatial(1024, ax1_0 * 512 + ax1_1 * 8 + ax1)
                                T.reads(
                                    B_local[v0, v1 // 2], Zeros[v0, 0], Scale[v0, 0]
                                )
                                T.writes(B_decode_local[v0, v1])
                                B_decode_local[v0, v1] = (
                                    T.Cast(
                                        "float16",
                                        T.bitwise_and(
                                            T.shift_right(
                                                B_local[v0, v1 // 2],
                                                T.Cast("int8", v1 % 2 * 4),
                                            ),
                                            T.int8(15),
                                        ),
                                    )
                                    - Zeros[v0, 0]
                                ) * Scale[v0, 0]
                        for ax0 in range(1):
                            for ax1 in T.vectorized(8):
                                with T.block("A_local"):
                                    v0 = T.axis.spatial(1, ax0)
                                    v1 = T.axis.spatial(
                                        1024, ax1_0 * 512 + ax1_1 * 8 + ax1
                                    )
                                    T.reads(A[v0, v1])
                                    T.writes(A_local[v0, v1])
                                    A_local[v0, v1] = A[v0, v1]
                        for ax1_2_0, ax1_2_1 in T.grid(4, 2):
                            with T.block("C"):
                                v0 = T.axis.spatial(1024, ax0_0 * 2 + ax0_1)
                                v1 = T.axis.reduce(
                                    1024,
                                    ax1_0 * 512 + ax1_1 * 8 + ax1_2_0 * 2 + ax1_2_1,
                                )
                                T.reads(A_local[0, v1], B_decode_local[v0, v1])
                                T.writes(C_local[0, v0])
                                with T.init():
                                    C_local[0, v0] = T.float16(0)
                                C_local[0, v0] = (
                                    C_local[0, v0]
                                    + A_local[0, v1] * B_decode_local[v0, v1]
                                )
                for ax0, ax1 in T.grid(1, 1):
                    with T.block("C_local"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(1024, ax0_0 * 2 + ax0_1 + ax1)
                        T.reads(C_local[v0, v1])
                        T.writes(D[0, v1])
                        D[0, v1] = C_local[v0, v1]

target = tvm.target.Target("cuda")
with tvm.transform.PassContext():
    rt_mod = tvm.build(Module, target=target)

torch_tensors = []
input_tensor = torch.randn(1, 1024).half().cuda()
weight_tensor = torch.randint(-8, 8, (1024, 512), dtype=torch.int8).cuda()
scale_tensor = torch.randn(1024).half().cuda().reshape(-1, 1)
zero_tensor = torch.zeros(1024).half().cuda().reshape(-1, 1)
output_tensor = torch.empty(1, 1024).half().cuda()
torch_tensors.append(input_tensor)
torch_tensors.append(weight_tensor)
torch_tensors.append(scale_tensor)
torch_tensors.append(zero_tensor)
torch_tensors.append(output_tensor)


dlpack_tensorss = []
for _ in range(1000):
    dlpack_tensors = [
        to_dlpack(torch_tensor) for torch_tensor in torch_tensors
    ]
    dlpack_tensorss.append(dlpack_tensors)
def from_dlpack_function():
    for _ in range(1000):
        tvm_nd_array_tensors = [
            tvm.runtime.ndarray.from_dlpack(dlpack_tensor)
            for dlpack_tensor in dlpack_tensorss[_]
        ]
    
from_dlpack_function()
