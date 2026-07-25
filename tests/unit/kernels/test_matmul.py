import pytest
import triton.language as tl
import torch

from pallas.ops import matmul

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bench_dtype(dtype: torch.dtype, M=4096, N=4096, K=4096):
    a = torch.rand((M, K), device="cuda", dtype=dtype)
    b = torch.rand((K, N), device="cuda", dtype=dtype)
 
    c_triton = matmul(a, b)
    c_torch = torch.matmul(a, b)
 
    rtol, atol = (1e-2, 1e-2) if dtype == torch.float16 else (2e-2, 2e-2)
    torch.testing.assert_close(c_triton, c_torch, rtol=rtol, atol=atol)