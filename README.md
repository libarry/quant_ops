# FlatQuant 动态量化 CUDA 算子

这个项目实现了一个高性能的 CUDA 算子，用于执行 FlatQuant 动态量化，将浮点输入（fp16/bf16/fp32）量化为 int4 格式。

## 功能特性

- **多精度支持**: 支持 fp16、bf16、fp32 输入
- **动态量化**: 基于 FlatQuant 方法的 per-token 动态量化
- **存储选项**: 支持 int8 存储或 int32 打包存储
- **高性能**: CUDA 加速的 Kronecker 矩阵乘法和量化计算
- **精度验证**: 提供与 PyTorch 参考实现的精度对比
- **性能基准**: 详细的性能测试和内存使用统计

## 系统要求

- CUDA 11.0+
- PyTorch 1.12.0+
- Python 3.7+
- NVIDIA GPU (计算能力 6.0+)

## 编译安装

### 方法 1: 使用构建脚本 (推荐)

```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

### 方法 2: 手动编译

```bash
# 设置环境变量
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
export CUDA_HOME=/usr/local/cuda

# 编译扩展
python setup.py build_ext --inplace
```

## 使用方法

### 基础用法

```python
import torch
import sys
sys.path.insert(0, 'src')  # 如果从源码运行
from quant_ops import flatquant_dynamic_quantize, dequantize_int4, get_decompose_dim

# 创建测试数据
batch_tokens = 128
features = 4096
M, N = get_decompose_dim(features)

input_tensor = torch.randn(batch_tokens, features, dtype=torch.float16, device='cuda')
left_trans = torch.randn(M, M, dtype=torch.float16, device='cuda')
right_trans = torch.randn(N, N, dtype=torch.float16, device='cuda')

# 执行量化
quantized, scales = flatquant_dynamic_quantize(
    input_tensor, 
    left_trans, 
    right_trans, 
    clip_ratio=1.0,
    pack_int32=False,  # 使用 int8 存储
    use_cuda=True      # 使用 CUDA 加速
)

# 反量化
dequantized = dequantize_int4(quantized, scales, packed_int32=False)

print(f"输入形状: {input_tensor.shape}")
print(f"量化结果形状: {quantized.shape}")
print(f"量化数据类型: {quantized.dtype}")
print(f"Scale 形状: {scales.shape}")
```

### 使用 int32 打包

```python
# 注意：features 必须是 8 的倍数
quantized_packed, scales = flatquant_dynamic_quantize(
    input_tensor,
    left_trans,
    right_trans,
    clip_ratio=1.0,
    pack_int32=True,   # 使用 int32 打包存储
    use_cuda=True
)

# 反量化打包的数据
dequantized_packed = dequantize_int4(quantized_packed, scales, packed_int32=True)

print(f"打包量化结果形状: {quantized_packed.shape}")  # [batch_tokens, features//8]
print(f"打包数据类型: {quantized_packed.dtype}")     # torch.int32
```

## 性能测试

### 运行精度测试

```bash
python tests/test_accuracy.py
```

### 运行性能测试

```bash
python tests/test_performance.py
```

### 运行所有测试

```bash
python tests/test_performance.py --test all
```

### 只运行精度测试

```bash
python tests/test_performance.py --test accuracy
```

### 只运行性能测试

```bash
python tests/test_performance.py --test performance
```

### 详细测试（包含不同数据类型）

```bash
python tests/test_performance.py --detailed
```

## API 参考

### `flatquant_dynamic_quantize`

执行 FlatQuant 动态量化到 int4。

**参数:**
- `input_tensor` (torch.Tensor): 输入张量 [batch_tokens, features]，支持 fp16/bf16/fp32
- `left_trans` (torch.Tensor): 左变换矩阵 [M, M]
- `right_trans` (torch.Tensor): 右变换矩阵 [N, N]，其中 M*N = features
- `clip_ratio` (float): 量化截断比例，默认 1.0
- `pack_int32` (bool): 是否打包为 int32，默认 False
- `use_cuda` (bool): 是否使用 CUDA 加速，默认 True

**返回:**
- `Tuple[torch.Tensor, torch.Tensor]`: (量化结果, scale值)

### `dequantize_int4`

反量化 int4 数据回浮点数。

**参数:**
- `quantized` (torch.Tensor): 量化后的数据
- `scales` (torch.Tensor): 量化 scale 值
- `packed_int32` (bool): 输入是否为 int32 打包格式，默认 False

**返回:**
- `torch.Tensor`: 反量化后的浮点张量

### `get_decompose_dim`

计算 FlatQuant 变换矩阵的分解维度。

**参数:**
- `n` (int): 特征维度

**返回:**
- `Tuple[int, int]`: (M, N) 使得 M*N = n

## 算法原理

这个 CUDA 算子实现了 FlatQuant 动态量化的核心步骤：

1. **Kronecker 矩阵乘法**: 使用左右变换矩阵对输入进行预处理，平滑激活分布
2. **动态量化参数计算**: 计算 per-token 的量化 scale 值
3. **量化**: 将浮点值量化到 int4 范围 [-8, 7]
4. **可选打包**: 将 8 个 int4 值打包到 1 个 int32 中以提高存储效率

## 性能特点

- **内存效率**: int32 打包可节省 50% 存储空间
- **计算加速**: CUDA 并行化的矩阵运算和量化计算
- **精度保持**: 与 PyTorch 参考实现保持高精度一致性

## 故障排除

### 编译错误

1. **CUDA 版本不匹配**: 确保 PyTorch CUDA 版本与系统 CUDA 版本兼容
2. **计算能力不支持**: 检查 GPU 计算能力是否 >= 6.0
3. **内存不足**: 降低 batch_tokens 或 features 大小

### 运行时错误

1. **维度不匹配**: 确保 M*N = features
2. **打包要求**: 使用 pack_int32=True 时，features 必须是 8 的倍数
3. **设备不匹配**: 确保所有张量都在 CUDA 设备上

## 许可证

本项目基于 Apache 2.0 许可证开源。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！ 