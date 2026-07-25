import pytest
import triton.language as tl
import torch

from pallas.ops import matmul

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bench_dtype(dtype: torch.dtype, M=4096, N=4096, K=4096):
    import triton.testing
 
    a = torch.rand((M, K), device="cuda", dtype=dtype)
    b = torch.rand((K, N), device="cuda", dtype=dtype)
 
    ms_triton = triton.testing.do_bench(lambda: matmul(a, b))
    ms_torch = triton.testing.do_bench(lambda: torch.matmul(a, b))
    flops = 2 * M * N * K
 
    print(f"\n--- dtype={dtype} shape=({M},{K})x({K},{N}) ---")
    print(f"Triton: {ms_triton*1000:.2f} us  ({flops/(ms_triton*1e-3)/1e12:.2f} TFLOPS)")
    print(f"Torch (cuBLAS): {ms_torch*1000:.2f} us  ({flops/(ms_torch*1e-3)/1e12:.2f} TFLOPS)")
 
 