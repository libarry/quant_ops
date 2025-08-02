# 项目目录结构

这个文档描述了重新整理后的标准化Python库目录结构。

## 目录结构

```
quant_ops/
├── src/                                    # 源代码目录
│   └── quant_ops/                         # 主包
│       ├── __init__.py                    # 包初始化，提供统一API
│       ├── ops/                           # 算子模块
│       │   ├── __init__.py
│       │   └── quantization/              # 量化算子
│       │       ├── __init__.py
│       │       └── flatquant/             # FlatQuant算法
│       │           ├── __init__.py
│       │           ├── ops.py             # 主要算子实现
│       │           └── kernels/           # CUDA内核
│       │               ├── __init__.py
│       │               └── cuda_quant_kernel.cu
│       └── utils/                         # 工具模块
│           ├── __init__.py
│           └── decompose.py               # 矩阵分解工具
├── tests/                                 # 测试模块
│   ├── __init__.py
│   ├── test_accuracy.py                   # 精度测试
│   └── test_performance.py               # 性能测试
├── scripts/                               # 脚本目录
│   └── build.sh                          # 构建脚本
├── examples/                              # 示例代码
│   └── basic_usage.py                     # 基础使用示例
├── setup.py                              # 构建配置（传统）
├── pyproject.toml                        # 项目配置（现代）
├── MANIFEST.in                           # 分发文件清单
├── README.md                             # 项目说明
└── w4a4_flatquant_dynamic.py            # 原有文件（可忽略）
```

## 模块说明

### 核心模块

- **src/quant_ops/__init__.py**: 主包入口，提供统一的API接口
- **src/quant_ops/ops/quantization/flatquant/ops.py**: FlatQuant算子的主要实现
- **src/quant_ops/utils/decompose.py**: 矩阵分解等工具函数

### CUDA内核

- **src/quant_ops/ops/quantization/flatquant/kernels/cuda_quant_kernel.cu**: CUDA内核实现

### 测试模块

- **tests/test_accuracy.py**: 精度测试，对比CUDA实现与PyTorch参考实现
- **tests/test_performance.py**: 性能基准测试，测试不同配置下的性能

### 构建和配置

- **setup.py**: 传统的setuptools配置，包含CUDA扩展编译
- **pyproject.toml**: 现代Python项目配置
- **scripts/build.sh**: 自动化构建脚本

## 使用方法

### 从源码安装

```bash
# 编译CUDA扩展
./scripts/build.sh

# 或手动编译
python setup.py build_ext --inplace
```

### 导入使用

```python
import sys
sys.path.insert(0, 'src')  # 如果从源码运行

from quant_ops import flatquant_dynamic_quantize, dequantize_int4, get_decompose_dim
```

### 运行测试

```bash
# 精度测试
python tests/test_accuracy.py

# 性能测试  
python tests/test_performance.py

# 示例
python examples/basic_usage.py
```

## 扩展新算子

要添加新的量化算子，建议按照以下结构：

```
src/quant_ops/ops/quantization/
├── flatquant/          # 现有FlatQuant算子
├── new_algorithm/      # 新算法目录
│   ├── __init__.py
│   ├── ops.py         # 算子实现
│   └── kernels/       # CUDA内核（如有）
│       ├── __init__.py
│       └── cuda_kernels.cu
```

然后在相应的 `__init__.py` 文件中添加导入，并在主包的 `__init__.py` 中暴露新的API。

## 优势

1. **模块化设计**: 每个算法独立组织，易于维护和扩展
2. **标准结构**: 遵循Python包的最佳实践
3. **测试分离**: 精度和性能测试分开，便于针对性测试
4. **文档完整**: 每个模块都有清晰的文档说明
5. **易于分发**: 支持标准的pip安装流程 