import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(  input_ptr, 
                            denominator_ptr,
                            max_ptr, 
                            input_row_stride, 
                            m, 
                            n, 
                            BLOCK_SIZE_M: tl.constexpr,
                            BLOCK_SIZE_N: tl.constexpr,
                            num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    m_offset = row_start * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    acc_denominator = tl.full((BLOCK_SIZE_M,), 0.0, dtype=tl.float32)
    acc_max = tl.full((BLOCK_SIZE_M,), -tl.inf, dtype=tl.float16)
    for row_idx in tl.range(0, tl.cdiv(n, BLOCK_SIZE_N),  num_stages=num_stages):
        n_offset = row_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = m_offset[:, None] < m and n_offset[None, :] < n
        inp_tile = tl.load(input_ptr + m_offset[:, None] * input_row_stride + n_offset[None, :], 
                    mask=mask, other=-tl.inf)
        tile_max = tl.max(inp_tile, axis=-1)
        tile_max = tl.maximum(acc_max, tile_max)
        stabilized_scores = inp_tile - tile_max[..., None]
        exp_scores = stabilized_scores.exp()
        acc_denominator = acc_denominator * (acc_max - tile_max).exp() + exp_scores.sum(axis=-1)
        acc_max = tile_max
    
    tl.store(denominator_ptr + row_start * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), acc_denominator)
    tl.store(max_ptr + row_start * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), acc_max)


def softmax(x: torch.Tensor):
    assert x.is_cuda
    denominator = torch.empty_like(x[:, 0])
    x_max = torch.empty_like(x[:, 0])
    m, n = x.shape

    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_M']), )
    online_softmax_kernel[grid](x, 
                                denominator, 
                                x_max, 
                                x.stride(0)
                                m,
                                n, 
                                BLOCK_SIZE_M=32, 
                                BLOCK_SIZE_N=32, 
                                num_stages=1)
    output = (x - x_max[:, None]).exp() / denominator[:, None]
    return output


if __name__ == "__main__":
    x = torch.randn(128, 128).cuda()
    output_triton = softmax(x)
    output_torch = torch.softmax(x, dim=-1)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')