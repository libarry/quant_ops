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

# 解析命令行参数
COMPILE_ALL=false
if [ "$1" == "--all" ]; then
    COMPILE_ALL=true
fi

# 设置环境变量
if [ "$COMPILE_ALL" = true ]; then
    echo "检测到 --all 参数，将为所有支持的架构编译。"
    export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
elif [ -n "$TORCH_CUDA_ARCH_LIST" ]; then
    echo "使用环境变量中已定义的 TORCH_CUDA_ARCH_LIST."
else
    echo "将自动检测当前 GPU 架构..."
    # 尝试检测架构，并将错误输出重定向，避免干扰
    CUDA_ARCH=$(python3 -c "import torch; major, minor = torch.cuda.get_device_capability(); print(f'{major}.{minor}')" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$CUDA_ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
        echo "成功检测到架构: $CUDA_ARCH"
    else
        echo "警告: 无法自动检测 GPU 架构。将使用默认列表进行编译。"
        export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
    fi
fi
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
    
else
    echo "构建失败！"
    exit 1
fi 