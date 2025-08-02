from setuptools import setup, find_packages
from torch.utils import cpp_extension
import torch
import os

# 检查 CUDA 是否可用并配置扩展
ext_modules = []
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA detected, building Int4 MatMul with CUDA support...")
    
    # CUDA 编译选项
    cuda_flags = [
        "-O3", 
        "-std=c++17",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        "-use_fast_math",
        "--ptxas-options=-v"
    ]

    # C++ 编译选项  
    cxx_flags = [
        "-O3", 
        "-std=c++17"
    ]

    # Int4 MatMul CUDA 内核文件路径
    int4_matmul_kernel_path = "src/quant_ops/ops/quantization/int4_matmul/kernels/cuda_int4_matmul_kernel.cu"

    # 只定义 Int4 MatMul 扩展模块
    ext_modules = [
        cpp_extension.CUDAExtension(
            name="quant_ops.ops.quantization.int4_matmul.kernels.cuda_int4_matmul_ops",
            sources=[int4_matmul_kernel_path],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": cuda_flags
            },
            include_dirs=cpp_extension.include_paths(),
            libraries=['cublas', 'curand'],
        )
    ]
else:
    print("CUDA not detected, cannot build Int4 MatMul CUDA extension...")

setup(
    name="quant-ops-int4",
    version="0.1.0", 
    author="Quant Ops Team",
    author_email="",
    description="Int4 Matrix Multiplication CUDA算子",
    
    # 包配置
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # 扩展模块
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    
    # Python要求
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.0",
        "pybind11>=2.6.0"
    ],
    
    zip_safe=False,
)