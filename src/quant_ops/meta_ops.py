from torch._ops import op
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