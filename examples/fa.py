import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def fast_attention_kernel(  Q, K, V, 
                            output, 
                            seq, 
                            hidden_dim,
                            q_stride, k_stride, v_stride, output_stride,
                            BLOCK_SIZE_HIDDEN: tl.constexpr,
                            BLOCK_SIZE_Q: tl.constexpr,
                            BLOCK_SIZE_KV: tl.constexpr,
                            num_stages: tl.constexpr):
    row_start = tl.program_id(0)

    q_offset = row_start * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    hidden_offset = tl.arange(0, BLOCK_SIZE_HIDDEN)

    acc_denominator = tl.full((BLOCK_SIZE_Q,), 0.0, dtype=tl.float32)
    acc_max = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    output_tile = tl.full((BLOCK_SIZE_Q, hidden_dim), 0.0, dtype=tl.float32)
    for row_idx in tl.range(0, tl.cdiv(seq, BLOCK_SIZE_KV), num_stages=num_stages):
        k_offset = row_idx * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)
        q_tile = tl.load(Q + q_offset[:, None] * q_stride + hidden_offset[None, :], 
            mask=q_offset[:, None] < seq and hidden_offset[None, :] < hidden_dim, other=0.0)
        k_tile = tl.load(K + k_offset[:, None] * k_stride + hidden_offset[None, :], 
            mask=k_offset[:, None] < seq and hidden_offset[None, :] < hidden_dim, other=0.0)
        v_tile = tl.load(V + k_offset[:, None] * v_stride + hidden_offset[None, :], 
            mask=q_offset[:, None] < seq and hidden_offset[None, :] < hidden_dim, other=0.0)
        score = tl.dot(q_tile, k_tile.transpose(-2, -1)) / hidden_dim ** 0.5

        current_max = tl.max(score, axis=-1)
        new_max = tl.maximum(acc_max, current_max)
        score = score - new_max[:, None]
        exp_score = tl.exp(score)
        current_denominator = acc_denominator * (acc_max - new_max).exp() + exp_score.sum(axis=-1)
        scale_factor = (acc_denominator / current_denominator)
        max_adjustment = (acc_max - new_max).exp()
        output_tile = output_tile * scale_factor[:, None] * max_adjustment[:, None] + (exp_score @ v_tile[:, :hidden_dim]) / current_denominator[:, None]
        acc_max = new_max
        acc_denominator = current_denominator

    tl.store(   output + q_offset[:, None] * output_stride + hidden_offset[None, :], output_tile)


def fast_attention(Q, K, V):
    assert Q.is_cuda
    bz, seq, hidden_dim = Q.shape
    output = torch.empty((bz, seq, hidden_dim), device=Q.device)
    grid = lambda meta: (triton.cdiv(seq, meta['BLOCK_SIZE_Q']), )
    for i in range(bz):
        q = Q[i]
        k = K[i]
        v = V[i]
        o = output[i]
        fast_attention_kernel[grid](q, 
                                    k, 
                                    v, 
                                    o, 
                                    seq, 
                                    hidden_dim, 
                                    q.stride(0), 
                                    k.stride(0), 
                                    v.stride(0), 
                                    o.stride(0), 
                                    BLOCK_SIZE_Q=32, 
                                    BLOCK_SIZE_KV=32, 
                                    num_stages=1)
    return output

def test_attention_implementations():
    """
    测试三种attention实现的正确性和一致性
    """
    print("=" * 60)
    print("Attention实现测试")
    print("=" * 60)
    
    # 创建测试数据
    batch_size, seq_len, embed_dim = 2, 128, 64
    query = torch.randn(batch_size, seq_len, embed_dim).cuda()
    key = value = query.clone().cuda()  # 简化测试
    
    print(f"测试数据形状: query{query.shape}, key{key.shape}, value{value.shape}")
    print()
    
    # 1. 使用PyTorch内置实现（参考基准）
    print("1. PyTorch内置实现 (参考基准)")
    pytorch_attn = F.scaled_dot_product_attention(query, key, value)
    print(f"   输出形状: {pytorch_attn.shape}")
    print()

    # 2. 朴素实现
    print("2. 朴素Scaled Dot-Product Attention")
    naive_attn = fast_attention(query, key, value)
    naive_diff = (pytorch_attn - naive_attn).abs().max()
    print(f"   输出形状: {naive_attn.shape}")
    print(f"   与基准最大差异: {naive_diff:.6f}")
    print()

if __name__ == "__main__":
    # 运行测试
    test_attention_implementations()