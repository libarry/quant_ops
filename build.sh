#!/bin/bash

echo "构建 FlatQuant CUDA 量化算子..."

# 检查 CUDA 是否可用
if ! command -v nvcc &> /dev/null; then
    echo "错误: nvcc 未找到。请确保 CUDA 工具包已安装并在 PATH 中。"
    exit 1
fi

# 检查 Python 和 PyTorch
python3 -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda}')" || {
    echo "错误: PyTorch 未正确安装或 CUDA 不可用。"
    exit 1
}

# 清理之前的构建
echo "清理之前的构建..."
rm -rf build/ dist/ *.egg-info/ cuda_quant_ops*.so

# 设置环境变量
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

# 构建扩展
echo "开始构建 CUDA 扩展..."
python3 setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "构建成功！"
    echo "生成的文件:"
    ls -la cuda_quant_ops*.so 2>/dev/null || echo "未找到 .so 文件"
    
    echo ""
    echo "运行测试..."
    python3 quant_ops.py
else
    echo "构建失败！"
    exit 1
fi 