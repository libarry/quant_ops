# Int4 矩阵乘法 CUDA 算子

这个模块实现了一个高性能的int4矩阵乘法CUDA算子，支持pack成int32的int4 tensor的矩阵乘法运算，并包含per-channel反量化功能。

## 功能特性

- **CUDA加速**: 使用CUDA实现的高性能int4矩阵乘法
- **内存高效**: 支持int32打包的int4数据格式，节省内存带宽
- **Per-channel反量化**: 支持矩阵A的per-row和矩阵B的per-column scale
- **PyTorch兼容**: 提供PyTorch实现作为备选方案
- **自动优化**: 根据矩阵大小自动选择最优kernel

## 算子接口

### 主要函数

```python
from quant_ops import int4_matmul, pack_int4_to_int32, generate_test_data

# 自动选择最优实现 (CUDA优先)
result = int4_matmul(a_packed, b_packed, scale_a, scale_b)

# 显式使用CUDA实现
result = int4_matmul_cuda(a_packed, b_packed, scale_a, scale_b)

# 显式使用PyTorch实现
result = int4_matmul_pytorch(a_packed, b_packed, scale_a, scale_b)
```

### 参数说明

- `a_packed`: 矩阵A，形状为 `[M, K//8]`，dtype为int32，每个int32包含8个int4值
- `b_packed`: 矩阵B，形状为 `[K//8, N]`，dtype为int32，每个int32包含8个int4值  
- `scale_a`: 矩阵A的per-row scale，形状为 `[M]`，dtype为float32
- `scale_b`: 矩阵B的per-column scale，形状为 `[N]`，dtype为float32
- 返回值: 反量化后的矩阵乘法结果，形状为 `[M, N]`，dtype为float16

### 数据格式

Int4值范围为 `[-8, 7]`，在int32中的打包格式：
```
int32_value = val0 + (val1 << 4) + (val2 << 8) + ... + (val7 << 28)
其中 val_i ∈ [0, 15] (对应 int4 ∈ [-8, 7])
```

## 构建和使用

### 1. 构建CUDA扩展

```bash
# 使用专门的构建脚本
./scripts/build_int4_matmul.sh

# 或者使用通用构建脚本
./scripts/build.sh --all
```

### 2. 运行基准测试

```bash
python examples/int4_matmul_benchmark.py
```

### 3. 基本使用示例

```python
import torch
from quant_ops import generate_test_data, int4_matmul

# 生成测试数据
M, K, N = 1024, 1024, 1024
a_packed, b_packed, scale_a, scale_b, a_original, b_original = generate_test_data(M, K, N)

# 执行int4矩阵乘法
result = int4_matmul(a_packed, b_packed, scale_a, scale_b)

print(f"输入A: {a_packed.shape} ({a_packed.dtype})")
print(f"输入B: {b_packed.shape} ({b_packed.dtype})")  
print(f"输出:  {result.shape} ({result.dtype})")
```

## 性能特性

### 优化策略

1. **内存访问优化**: 使用shared memory缓存tile数据
2. **计算优化**: 循环展开和向量化计算
3. **自适应kernel**: 根据矩阵大小选择最优策略
4. **内存带宽**: int4打包减少内存访问量

### 性能预期

相比于传统的fp16矩阵乘法：
- **内存使用**: 减少约4倍（int4 vs fp16）
- **计算速度**: 在大矩阵上可能有1.5-3x加速
- **精度损失**: 最小，适合大多数量化推理场景

## 文件结构

```
src/quant_ops/ops/quantization/int4_matmul/
├── __init__.py                 # 模块初始化
├── ops.py                      # Python接口和PyTorch实现
└── kernels/
    ├── __init__.py             # Kernel模块初始化
    └── cuda_int4_matmul_kernel.cu  # CUDA kernel实现

examples/
└── int4_matmul_benchmark.py    # 性能和精度基准测试

scripts/
└── build_int4_matmul.sh        # 专用构建脚本
```

## 技术细节

### CUDA Kernel设计

1. **简单kernel**: 每个thread处理一个输出元素，适合小矩阵
2. **优化kernel**: 使用16x16 tile的shared memory，适合大矩阵
3. **Unpack操作**: 在kernel内部高效解包int32到int4
4. **Scale应用**: 在计算最后阶段应用per-channel scale

### 数值精度

- Int4量化范围: [-8, 7] 
- 中间计算使用float32累加器避免溢出
- 最终结果转换为float16节省内存
- Per-channel scale提供细粒度的量化控制

### 错误处理

- 自动检测CUDA可用性
- 优雅降级到PyTorch实现
- 详细的错误信息和调试支持
- 输入参数验证和形状检查