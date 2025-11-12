import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def fast_attention_kernel(  Q, 
                            K, 
                            V, 
                            output, 
                            sqrt_hidden_dim,
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
    output_tile = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_HIDDEN), 0.0, dtype=tl.float32)
    for row_idx in tl.range(0, tl.cdiv(seq, BLOCK_SIZE_KV), num_stages=num_stages):
        k_offset = row_idx * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)
        q_tile = tl.load(
            Q + q_offset[:, None] * q_stride + hidden_offset[None, :],
            mask=(q_offset[:, None] < seq) & (hidden_offset[None, :] < hidden_dim),
            other=0
        ).to(tl.float32)
        k_tile = tl.load(
            K + k_offset[:, None] * k_stride + hidden_offset[None, :],
            mask=(k_offset[:, None] < seq) & (hidden_offset[None, :] < hidden_dim),
            other=0
        ).to(tl.float32)
        v_tile = tl.load(
            V + k_offset[:, None] * v_stride + hidden_offset[None, :],
            mask=(k_offset[:, None] < seq) & (hidden_offset[None, :] < hidden_dim),
            other=0
        ).to(tl.float32)
        score = tl.dot(q_tile, tl.trans(k_tile)) 
        scale = tl.full((1,), sqrt_hidden_dim, dtype=tl.float32)
        score = score / scale
        # 对越界的KV列进行softmax掩码：将分数置为 -inf，避免将padding列当作有效token参与max和sum
        kv_valid = k_offset[None, :] < seq
        neg_inf = tl.full((1,), float("-inf"), dtype=tl.float32)
        masked_score = tl.where(kv_valid, score, neg_inf)

        current_max = tl.max(masked_score, axis=-1)
        new_max = tl.maximum(acc_max, current_max)
        score = masked_score - new_max[:, None]
        exp_score = tl.exp(score)
        current_denominator = acc_denominator * (acc_max - new_max).exp() + exp_score.sum(axis=-1)
        scale_factor = (acc_denominator / current_denominator)
        max_adjustment = (acc_max - new_max).exp()
        output_tile = output_tile * scale_factor[:, None] * max_adjustment[:, None] + tl.dot(exp_score,  v_tile) / current_denominator[:, None]
        acc_max = new_max
        acc_denominator = current_denominator

    tl.store(
        output + q_offset[:, None] * output_stride + hidden_offset[None, :],
        output_tile,
        mask=(q_offset[:, None] < seq) & (hidden_offset[None, :] < hidden_dim)
    )


def fast_attention(Q, K, V):
    assert Q.is_cuda
    bz, seq, hidden_dim = Q.shape
    output = torch.empty((bz, seq, hidden_dim), device=Q.device, dtype=Q.dtype)
    grid = lambda meta: (triton.cdiv(seq, meta['BLOCK_SIZE_Q']), )
    sqrt_hidden_dim = hidden_dim ** 0.5
    for i in range(bz):
        q = Q[i]
        k = K[i]
        v = V[i]
        o = output[i]
        fast_attention_kernel[grid](q, 
                                    k, 
                                    v, 
                                    o, 
                                    sqrt_hidden_dim,
                                    seq, 
                                    hidden_dim, 
                                    q.stride(0), 
                                    k.stride(0), 
                                    v.stride(0), 
                                    o.stride(0), 
                                    BLOCK_SIZE_HIDDEN=512,
                                    BLOCK_SIZE_Q=32, 
                                    BLOCK_SIZE_KV=32, 
                                    num_stages=1)
    return output

def flash_attention_v1_fake(query, key, value, mask=None, tile_size=32):
    """
    Flash Attention V1实现
    完全在线计算，避免存储任何中间attention矩阵
    最高效的内存使用方式
    
    Args:
        query: [batch_size, seq_len, embed_dim]
        key: [batch_size, seq_len, embed_dim]
        value: [batch_size, seq_len, embed_dim]
        mask: 可选的attention mask
        tile_size: 分块大小
        
    Returns:
        attention_output: [batch_size, seq_len, embed_dim]
    """
    key_dim = query.size(-1)
    batch_size, seq_len, embed_dim = query.size()
    
    # 初始化累积变量
    cumulative_denominator = torch.zeros(batch_size, seq_len, device=query.device)
    cumulative_max = torch.full((batch_size, seq_len), -torch.inf, device=query.device)
    
    # 初始化输出张量
    attention_output = torch.zeros(batch_size, seq_len, embed_dim, device=query.device)
    
    # 分块处理：完全在线计算
    for chunk_start in range(0, seq_len, tile_size):
        chunk_end = chunk_start + tile_size
        
        # 提取当前分块的key和value
        key_chunk = key[..., chunk_start:chunk_end, :].clone()
        value_chunk = value[..., chunk_start:chunk_end, :].clone()
        
        # 计算当前分块的attention分数
        # Q @ K_chunk^T / sqrt(d_k)
        scores_chunk = query @ key_chunk.transpose(-2, -1) / key_dim ** 0.5
        
        # 计算当前分块的最大值
        chunk_max = scores_chunk.max(dim=-1, keepdim=False).values
        
        # 更新全局最大值
        new_max = torch.maximum(cumulative_max, chunk_max)
        
        # 数值稳定化并指数化
        stabilized_scores = scores_chunk - new_max.unsqueeze(-1)
        exp_scores = stabilized_scores.exp()
        
        # 计算新的累积分母
        new_denominator = cumulative_denominator * (cumulative_max - new_max).exp() + \
                         exp_scores.sum(dim=-1, keepdim=False)
        
        # 更新输出：在线累积attention结果
        # 公式推导：
        # output_new = (output_old * d_old * exp(m_old - m_new) + score_chunk @ V_chunk) / d_new
        scale_factor_old = (cumulative_denominator / new_denominator).unsqueeze(-1)
        max_adjustment = (cumulative_max - new_max).exp().unsqueeze(-1)
        
        attention_output = attention_output * scale_factor_old * max_adjustment + \
                          (exp_scores @ value_chunk.float()).to(attention_output.dtype) / new_denominator.unsqueeze(-1)
        
        # 更新累积变量
        cumulative_max = new_max
        cumulative_denominator = new_denominator
    
    return attention_output



def test_attention_implementations():
    """
    测试三种attention实现的正确性和一致性
    """
    print("=" * 60)
    print("Attention实现测试")
    print("=" * 60)
    
    # 创建测试数据
    batch_size, seq_len, embed_dim = 2, 1000, 256
    query = torch.randn(batch_size, seq_len, embed_dim).cuda().to(torch.float16)
    key = value = query.clone().cuda().to(torch.float16)  # 简化测试
    
    print(f"测试数据形状: query{query.shape}, key{key.shape}, value{value.shape}")
    print()
    
    # 1. 使用PyTorch内置实现（参考基准）
    print("1. PyTorch内置实现 (参考基准)")
    pytorch_attn = F.scaled_dot_product_attention(query, key, value)
    print(f"   输出形状: {pytorch_attn.shape}")
    print()

    # 2. triton实现
    print("2. triton Attention")
    naive_attn = fast_attention(query, key, value)
    naive_diff = (pytorch_attn - naive_attn).abs().max()
    print(f"   输出形状: {naive_attn.shape}")
    print(f"   与基准最大差异: {naive_diff:.6f}")
    print()

    # 4. Flash Attention V1 pseudo实现
    print("4. Flash Attention V1 pseudo实现")
    flash_attn = flash_attention_v1_fake(query, key, value)
    flash_diff = (pytorch_attn - flash_attn).abs().max()
    print(f"   输出形状: {flash_attn.shape}")
    print(f"   与基准最大差异: {flash_diff:.6f}")
    print()

@torch.inference_mode()
def benchmark_fast_attention(
    batch_sizes=(1,),
    # seq_list=(256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096),
    seq_list=(1000,2000),
    hidden_dims=(256,),
    dtype=torch.float16,
    warmup=25,
    rep=100,
    seed=0,
):
    """
    参考 Triton 矩阵乘法教程的基准设置，使用 CUDA events 进行计时：
    - warmup 次预热
    - rep 次重复，取平均时延
    - 打印每组形状下 PyTorch 和 Triton 的耗时与 TFLOPS 估算
    估算 FLOPs：约 4 * B * S^2 * H（QK^T 与 PV 两个 GEMM 的近似量，不含 softmax）
    """
    torch.manual_seed(seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    def bench_ms(fn):
        # warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        return total_ms / rep

    print("=" * 60)
    print("fast_attention vs. torch.nn.functional.scaled_dot_product_attention 基准测试")
    print(f"warmup={warmup}, rep={rep}, dtype={dtype}")
    print("=" * 60)
    header = f"{'B':>3} {'S':>6} {'H':>6}  {'PyTorch ms':>12} {'PyTorch TFLOPS':>16}  {'Triton ms':>10} {'Triton TFLOPS':>15}  {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for B in batch_sizes:
        for H in hidden_dims:
            for S in seq_list:
                # 构造输入
                q = torch.randn(B, S, H, device=device, dtype=dtype)
                k = q.clone()
                v = q.clone()

                # PyTorch baseline
                def fn_pt():
                    return F.scaled_dot_product_attention(q, k, v)

                # Triton
                def fn_triton():
                    return fast_attention(q, k, v)

                ms_pt = bench_ms(fn_pt)
                ms_triton = bench_ms(fn_triton)

                # 近似 FLOPs（忽略 softmax 和缩放）
                flops = 4.0 * B * (S ** 2) * H
                tflops_pt = (flops / 1e12) / (ms_pt / 1e3)
                tflops_triton = (flops / 1e12) / (ms_triton / 1e3)
                speedup = ms_pt / ms_triton if ms_triton > 0 else float("inf")

                print(f"{B:>3} {S:>6} {H:>6}  {ms_pt:12.3f} {tflops_pt:16.3f}  {ms_triton:10.3f} {tflops_triton:15.3f}  {speedup:8.2f}x")

if __name__ == "__main__":
    # 运行测试
    test_attention_implementations()
    # 如需运行基准，可取消下行注释
    benchmark_fast_attention()