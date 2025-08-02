# Quant Ops 安装指南

本指南将帮助您安装和配置 Quant Ops 量化算子库。

## 系统要求

### 必需组件
- Python 3.7+
- PyTorch 1.12.0+
- NVIDIA GPU (计算能力 6.0+) - 用于CUDA加速
- CUDA 11.0+ - 用于编译CUDA扩展

### 可选组件
- 没有CUDA环境也可以安装，但只能使用PyTorch后备实现

## 安装方法

### 方法 1: 使用 pip 安装（推荐）

```bash
# 确保在项目根目录
cd /path/to/quant_ops

# 直接安装
pip install .

# 或者以开发模式安装（修改代码后无需重新安装）
pip install -e .
```

### 方法 2: 从源码编译安装

```bash
# 1. 克隆或下载源码
git clone <repository-url>
cd quant_ops

# 2. 安装依赖
pip install torch pybind11

# 3. 编译并安装
python setup.py install
```

## 安装验证

安装完成后，运行以下命令验证安装：

```bash
# 基本验证
python -m quant_ops

# 完整测试（需要CUDA）
python -m quant_ops --test

# 查看版本
python -m quant_ops --version
```

## 使用示例

安装成功后，您可以在任何地方导入和使用：

```python
import torch
import quant_ops

# 基本使用
batch_tokens, features = 128, 4096
M, N = quant_ops.get_decompose_dim(features)

input_tensor = torch.randn(batch_tokens, features, dtype=torch.float16, device='cuda')
left_trans = torch.randn(M, M, dtype=torch.float16, device='cuda')
right_trans = torch.randn(N, N, dtype=torch.float16, device='cuda')

# 执行量化
quantized, scales = quant_ops.flatquant_dynamic_quantize(
    input_tensor, left_trans, right_trans, clip_ratio=1.0
)

# 反量化
dequantized = quant_ops.dequantize_int4(quantized, scales)

print(f"量化误差: {(input_tensor.float() - dequantized).abs().mean():.6f}")
```

## 故障排除

### 常见问题

1. **CUDA 编译错误**
   ```
   解决方案：
   - 确保 CUDA 工具包已正确安装
   - 检查 PyTorch CUDA 版本与系统 CUDA 版本兼容性
   - 设置正确的 CUDA_HOME 环境变量
   ```

2. **导入错误**
   ```python
   ImportError: No module named 'quant_ops'
   ```
   ```
   解决方案：
   - 确认安装成功：pip list | grep quant-ops
   - 检查 Python 环境是否正确
   - 尝试重新安装：pip uninstall quant-ops && pip install .
   ```

3. **CUDA 算子不可用**
   ```
   解决方案：
   - 检查 GPU 是否可用：torch.cuda.is_available()
   - 确认 CUDA 扩展编译成功
   - 查看安装日志确认没有编译错误
   ```

### 环境变量配置

```bash
# 设置 CUDA 路径（如果需要）
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置 PyTorch CUDA 架构（如果需要）
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
```

### 详细日志

如果安装时遇到问题，可以使用详细模式查看日志：

```bash
# 查看详细安装日志
pip install . -v

# 或者
python setup.py build_ext --inplace -v
```

## 卸载

```bash
pip uninstall quant-ops
```

## 获取帮助

如果遇到问题，请：

1. 检查系统要求和依赖
2. 查看故障排除部分
3. 运行 `python -m quant_ops` 进行诊断
4. 提交Issue并包含详细的错误信息和环境信息 