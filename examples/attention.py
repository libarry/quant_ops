import torch
import torch.nn.functional as F


# =============================================================================
# 朴素Attention实现 - 标准Scaled Dot-Product Attention
# =============================================================================
def naive_scaled_dot_product_attention(query, key, value, mask=None):
    """
    朴素实现的Scaled Dot-Product Attention
    使用标准的三步softmax计算，需要存储完整的attention矩阵
    
    Args:
        query: [batch_size, seq_len, embed_dim]
        key: [batch_size, seq_len, embed_dim]  
        value: [batch_size, seq_len, embed_dim]
        mask: 可选的attention mask
        
    Returns:
        attention_output: [batch_size, seq_len, embed_dim]
    """
    # 获取key的维度用于缩放
    key_dim = query.size(-1)
    
    # 1. 计算attention分数矩阵
    # Q @ K^T / sqrt(d_k)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / key_dim ** 0.5
    
    # 2. 应用mask（如果提供）
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    
    # 3. 计算softmax概率分布
    # 数值稳定性的softmax计算（三步法）
    # 步骤1: 减去最大值防止数值溢出
    max_scores = attention_scores.max(dim=-1, keepdim=True).values
    normalized_scores = attention_scores - max_scores
    
    # 步骤2: 指数化
    exp_scores = normalized_scores.exp()
    
    # 步骤3: 归一化
    attention_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
    
    # 4. 应用attention权重到value上
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_output


# =============================================================================
# Online Softmax Attention实现 - 分块计算softmax
# =============================================================================
def online_softmax_attention(query, key, value, mask=None, tile_size=32):
    """
    Online Softmax Attention实现
    使用分块计算softmax，避免存储完整的attention矩阵
    适用于内存受限的场景
    
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
    
    # 1. 计算完整的attention分数矩阵
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / key_dim ** 0.5
    
    # 2. 应用mask（如果提供）
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    
    # 3. 初始化online softmax的累积变量
    batch_size, seq_len = attention_scores.size(0), attention_scores.size(1)
    
    # 累积分母（用于softmax归一化）
    cumulative_denominator = torch.zeros(batch_size, seq_len, device=attention_scores.device)
    
    # 累积最大值（用于数值稳定性）
    cumulative_max = torch.full((batch_size, seq_len), -torch.inf, device=attention_scores.device)
    
    # 4. 分块处理：online softmax计算
    for chunk_start in range(0, attention_scores.size(-1), tile_size):
        chunk_end = chunk_start + tile_size
        
        # 提取当前分块的scores
        scores_chunk = attention_scores[..., chunk_start:chunk_end].clone()
        
        # 计算当前分块的最大值
        chunk_max = scores_chunk.max(dim=-1, keepdim=False).values
        
        # 更新全局最大值
        new_max = torch.maximum(cumulative_max, chunk_max)
        
        # 数值稳定化：减去新最大值
        stabilized_scores = scores_chunk - new_max.unsqueeze(-1)
        
        # 指数化
        exp_scores = stabilized_scores.exp()
        
        # 更新累积分母
        # 公式: d_sum_new = d_sum_old * exp(old_max - new_max) + sum(exp_scores_chunk)
        cumulative_denominator = cumulative_denominator * (cumulative_max - new_max).exp() + \
                                exp_scores.sum(dim=-1, keepdim=False)
        
        # 更新累积最大值
        cumulative_max = new_max
    
    # 5. 计算最终的attention权重
    # 使用累积的最大值和分母计算softmax
    stabilized_all_scores = attention_scores - cumulative_max.unsqueeze(-1)
    attention_weights = stabilized_all_scores.exp() / cumulative_denominator.unsqueeze(-1)
    
    # 6. 应用attention权重
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_output


# =============================================================================
# Flash Attention V1实现 - 完全在线计算
# =============================================================================
def flash_attention_v1(query, key, value, mask=None, tile_size=32):
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
                          (exp_scores @ value_chunk) / new_denominator.unsqueeze(-1)
        
        # 更新累积变量
        cumulative_max = new_max
        cumulative_denominator = new_denominator
    
    return attention_output


# =============================================================================
# 测试和验证代码
# =============================================================================
def test_attention_implementations():
    """
    测试三种attention实现的正确性和一致性
    """
    print("=" * 60)
    print("Attention实现测试")
    print("=" * 60)
    
    # 创建测试数据
    batch_size, seq_len, embed_dim = 2, 128, 64
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = value = query  # 简化测试
    
    print(f"测试数据形状: query{query.shape}, key{key.shape}, value{value.shape}")
    print()
    
    # 1. 使用PyTorch内置实现（参考基准）
    print("1. PyTorch内置实现 (参考基准)")
    pytorch_attn = F.scaled_dot_product_attention(query, key, value)
    print(f"   输出形状: {pytorch_attn.shape}")
    print()
    
    # 2. 朴素实现
    print("2. 朴素Scaled Dot-Product Attention")
    naive_attn = naive_scaled_dot_product_attention(query, key, value)
    naive_diff = (pytorch_attn - naive_attn).abs().max()
    print(f"   输出形状: {naive_attn.shape}")
    print(f"   与基准最大差异: {naive_diff:.6f}")
    print()
    
    # 3. Online Softmax实现
    print("3. Online Softmax Attention")
    online_attn = online_softmax_attention(query, key, value)
    online_diff = (pytorch_attn - online_attn).abs().max()
    print(f"   输出形状: {online_attn.shape}")
    print(f"   与基准最大差异: {online_diff:.6f}")
    print()
    
    # 4. Flash Attention V1实现
    print("4. Flash Attention V1")
    flash_attn = flash_attention_v1(query, key, value)
    flash_diff = (pytorch_attn - flash_attn).abs().max()
    print(f"   输出形状: {flash_attn.shape}")
    print(f"   与基准最大差异: {flash_diff:.6f}")
    print()
    
    # 总结
    print("=" * 60)
    print("测试总结:")
    print(f"   所有实现与基准的最大差异: {max(naive_diff, online_diff, flash_diff):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    test_attention_implementations()