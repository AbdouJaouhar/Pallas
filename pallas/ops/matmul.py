import triton
import triton.language as tl
import torch


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx = tl.program_id(axis=1)

    a_ptr += batch_idx * stride_ab
    b_ptr += batch_idx * stride_bb
    c_ptr += batch_idx * stride_cb

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = num_pid_n * GROUP_M
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M

    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)

def _launch_matmul(
    a, b, c, batch_size, M, K, N,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn
):
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        batch_size,
    )
    matmul_kernel[grid](
        a, b, c,
        M, K, N,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=32,
        BLOCK_SIZE_N=64,
        GROUP_M=8,
    )

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.device == b.device, "a and b should be on same device"
    assert a.dtype == b.dtype, "a and b should be of same type"

    if b.ndim == 2:
        *batch_dims, M_, K = a.shape
        K2, N = b.shape
        assert K == K2, f"K mismatch: {K} vs {K2}"

        a_flat = a.reshape(-1, K)
        M = a_flat.shape[0]
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        _launch_matmul(
            a_flat, b, c,
            batch_size=1,
            M=M, K=K, N=N,
            stride_ab=0, stride_am=a_flat.stride(0), stride_ak=a_flat.stride(1),
            stride_bb=0, stride_bk=b.stride(0), stride_bn=b.stride(1),
            stride_cb=0, stride_cm=c.stride(0), stride_cn=c.stride(1),
        )
        return c.reshape(*batch_dims, M_, N)

    assert a.ndim >= 3 and b.ndim >= 3, f"unsupported ranks: a{a.shape}, b{b.shape}"
    assert a.shape[:-2] == b.shape[:-2], f"batch dims must match: {a.shape[:-2]} vs {b.shape[:-2]}"

    *batch_dims, M, K = a.shape
    *_, K2, N = b.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"

    batch_size = 1
    for d in batch_dims:
        batch_size *= d

    a_flat = a.reshape(batch_size, M, K)
    b_flat = b.reshape(batch_size, K, N)
    c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)

    _launch_matmul(
        a_flat, b_flat, c,
        batch_size=batch_size,
        M=M, K=K, N=N,
        stride_ab=a_flat.stride(0), stride_am=a_flat.stride(1), stride_ak=a_flat.stride(2),
        stride_bb=b_flat.stride(0), stride_bk=b_flat.stride(1), stride_bn=b_flat.stride(2),
        stride_cb=c.stride(0), stride_cm=c.stride(1), stride_cn=c.stride(2),
    )
    return c.reshape(*batch_dims, M, N)