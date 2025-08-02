from setuptools import setup, find_packages
from torch.utils import cpp_extension
import torch
import os

# 检查 CUDA 是否可用
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This extension requires CUDA support.")

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

# CUDA 内核文件路径
cuda_kernel_path = "src/quant_ops/ops/quantization/flatquant/kernels/cuda_quant_kernel.cu"

# 定义扩展模块
ext_modules = [
    cpp_extension.CUDAExtension(
        name="cuda_quant_ops",
        sources=[cuda_kernel_path],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": cuda_flags
        },
        include_dirs=cpp_extension.include_paths(),
        libraries=['cublas', 'curand'],
    )
]

# 读取README作为长描述
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="quant-ops",
    version="0.1.0", 
    author="Quant Ops Team",
    author_email="",
    description="高性能量化算子库",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/quant-ops",
    
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
    
    # 分类器
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # 附加文件
    include_package_data=True,
    zip_safe=False,
    
    # 项目URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/quant-ops/issues",
        "Source": "https://github.com/your-org/quant-ops",
    },
) 