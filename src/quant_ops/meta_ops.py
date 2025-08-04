# 兼容 PyTorch >=2.3 无 torch._ops.op 的情况
try:
    from torch._ops import op  # PyTorch < 2.3
except ImportError:  # PyTorch 2.3+
    # 使用新版 torch.library.register_fake 作为 meta kernel 注册接口
    from torch.library import register_fake

    def op(name: str, kind: str = "meta"):
        """兼容装饰器，用于在缺失 torch._ops.op 时注册 meta kernel。"""
        assert kind == "meta", "Only meta registration is supported in this fallback"

        def decorator(fn):
            register_fake(name)(fn)
            return fn

        return decorator

import torch

# ---------------- Meta kernels for torch dispatcher -----------------

@op("quant_ops::int4_matmul", "meta")
def _int4_matmul_meta(a_packed: torch.Tensor, b_packed: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor):
    M = a_packed.size(0)
    N = b_packed.size(1)
    return a_packed.new_empty((M, N), dtype=torch.float16)


@op("quant_ops::flatquant_dynamic_quantize", "meta")
def _flatquant_meta(input_tensor: torch.Tensor, left_trans: torch.Tensor, right_trans: torch.Tensor, clip_ratio: float = 1.0, pack_int32: bool = False):
    batch_tokens, features = input_tensor.shape
    if pack_int32:
        out = input_tensor.new_empty((batch_tokens, features // 8), dtype=torch.int32)
    else:
        out = input_tensor.new_empty((batch_tokens, features), dtype=torch.int8)
    scale = input_tensor.new_empty((batch_tokens,), dtype=torch.float32)
    return (out, scale) 