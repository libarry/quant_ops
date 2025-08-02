#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math
import torch
import torch_npu
from typing import Any, Dict, Optional
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def pack_int4_weights_with_npu(weight_tensor):
    """
    使用npu_convert_weight_to_int4pack将权重打包为int4格式
    """
    try:
        org_device = weight_tensor.device
        weight_tensor = weight_tensor.npu()
        # 使用NPU专用的int4打包函数
        weight_int4_packed = torch_npu.npu_convert_weight_to_int4pack(
            weight_tensor.to(torch.int32),
            inner_k_tiles=1  # 默认值，可根据需要调整
        )
        return weight_int4_packed.to(org_device)
    except Exception as e:
        print(f"npu_convert_weight_to_int4pack不可用，使用手动打包: {e}")
        # fallback到手动实现
        return pack_int4_to_int32_manual(weight_tensor)


def pack_int4_to_int32_manual(int4_tensor):
    """
    手动将int4张量打包到int32中（备用方案）
    """
    # 确保最后一维是8的倍数
    assert int4_tensor.shape[-1] % 8 == 0, "最后一维必须是8的倍数"
    
    # 将int4值限制在[-8, 7]范围内
    int4_clamped = torch.clamp(int4_tensor, -8, 7)
    
    # 转换为uint4 [0, 15]
    uint4_tensor = int4_clamped + 8
    
    # 重塑并打包
    shape = list(uint4_tensor.shape)
    shape[-1] = shape[-1] // 8
    
    uint4_reshaped = uint4_tensor.view(*shape[:-1], -1, 8)
    
    # 打包到int32中
    packed = torch.zeros(*shape, dtype=torch.int32, device=uint4_tensor.device)
    for i in range(8):
        packed += (uint4_reshaped[..., i].to(torch.int32) << (i * 4))
    
    return packed


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


class AscendW4A4FlatQuantDynamicLinearMethod:
    """Linear method for Ascend W4A4_FLATQUANT_DYNAMIC.
    
    This class implements W4A4 quantization with FlatQuant approach and dynamic activation quantization.
    - Weight: 4-bit quantization (per-channel) with scale and offset, stored as int8 and packed to int32 during loading
    - Activation: 4-bit dynamic quantization with FlatQuant transform matrices (left_trans, right_trans) for distribution smoothing
    - Parameters: clip_ratio for controlling quantization clipping, weight_offset for asymmetric quantization, loaded from external weights
    """
    input_size = 0
    output_size = 0

    def __init__(self):
        self.transpose_weight = False
        self.sym = True  # 使用对称量化

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        """
        获取权重参数字典
        
        Args:
            input_size: 输入维度
            output_size: 输出维度
            params_dtype: 参数数据类型
            
        Returns:
            权重参数字典
        """
        # 确保输入维度是8的倍数，用于int4打包
        assert input_size % 8 == 0, f"input_size ({input_size}) must be divisible by 8 for int4 packing"
        AscendW4A4FlatQuantDynamicLinearMethod.input_size = input_size
        AscendW4A4FlatQuantDynamicLinearMethod.output_size = output_size
        params_dict = {
            # 原始int8保存的int4权重数据，未打包
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        """
        获取per-tensor量化参数字典
        
        Args:
            params_dtype: 参数数据类型
            
        Returns:
            per-tensor参数字典
        """
        params_dict = {}
        # FlatQuant变换矩阵（左变换和右变换）
        # 实际使用时从配置文件或权重文件中加载
        left_trans_dim, right_trans_dim = get_decompose_dim(AscendW4A4FlatQuantDynamicLinearMethod.input_size)
        params_dict["left_trans"] = torch.empty(left_trans_dim, left_trans_dim, dtype=params_dtype)
        params_dict["right_trans"] = torch.empty(right_trans_dim, right_trans_dim, dtype=params_dtype)
        
        # 量化截断比例参数
        params_dict["clip_ratio"] = torch.empty(1, dtype=torch.float32)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """
        获取per-channel量化参数字典
        
        Args:
            output_size: 输出维度
            params_dtype: 参数数据类型
            
        Returns:
            per-channel参数字典
        """
        params_dict = {}
        # 权重量化scale (per-channel)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        # 权重量化offset (per-channel)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=torch.float32)
    
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        应用W4A4 FlatQuant动态量化的前向传播
        
        Args:
            layer: 线性层模块
            x: 输入张量
            bias: 偏置项
            tp_rank: 张量并行rank
            
        Returns:
            输出张量
        """
        original_dtype = x.dtype
        input_shape = x.shape
        in_features = input_shape[-1]
        
        # 获取FlatQuant变换矩阵维度
        M = layer.left_trans.shape[0]
        N = layer.right_trans.shape[0]
        
        # 确保 M * N == in_features
        assert M * N == in_features, f"FlatQuant transform matrices dimension mismatch: M({M}) * N({N}) != in_features({in_features})"
        
        # 确保变换矩阵类型与输入匹配
        left_trans_matched = layer.left_trans.to(original_dtype)
        right_trans_matched = layer.right_trans.to(original_dtype)
        
        # 重塑输入：[batch_size * seq_len, M, N] 
        x_reshaped = x.view(-1, M, N)
        
        # 1. 使用npu_kronecker_quant进行激活值动态量化
        # npu_kronecker_quant算子一次最多处理8192个token，需要分批处理
        batch_tokens = x_reshaped.shape[0]
        max_batch_size = 8192
        
        if batch_tokens <= max_batch_size:
            # 不需要分批，直接处理
            x_quantized_int4, activation_scale = torch_npu.npu_kronecker_quant(
                x_reshaped,
                left_trans_matched, 
                right_trans_matched,
                clip_ratio=layer.aclnn_clip_ratio,
                dst_dtype=torch.int32
            )
        else:
            # 需要分批处理
            x_quantized_int4_list = []
            activation_scale_list = []
            
            for start_idx in range(0, batch_tokens, max_batch_size):
                end_idx = min(start_idx + max_batch_size, batch_tokens)
                x_batch = x_reshaped[start_idx:end_idx]
                
                x_quantized_batch, activation_scale_batch = torch_npu.npu_kronecker_quant(
                    x_batch,
                    left_trans_matched, 
                    right_trans_matched,
                    clip_ratio=layer.aclnn_clip_ratio,
                    dst_dtype=torch.int32
                )
                
                x_quantized_int4_list.append(x_quantized_batch)
                activation_scale_list.append(activation_scale_batch)
            
            # 合并分批处理的结果
            x_quantized_int4 = torch.cat(x_quantized_int4_list, dim=0)
            activation_scale = torch.cat(activation_scale_list, dim=0)
        
        # 2. 调用npu_quant_matmul进行int4矩阵乘法
        # 重塑量化后的激活值以适配matmul
        x_quantized_reshaped = x_quantized_int4.view(-1, M * N // 8)
        
        # npu_quant_matmul要求: pertoken_scale为float32，weight_scale为float32
        pertoken_scale = activation_scale.view(-1).to(torch.float32)
        
        output = torch_npu.npu_quant_matmul(
            x_quantized_reshaped,
            layer.weight_packed.t(),  # 转置打包的权重，已确保为int32
            layer.weight_scale.view(-1).to(torch.float32),  # 确保为float32
            pertoken_scale=pertoken_scale,  # 已确保为float32
            bias=None,
            output_dtype=original_dtype  # 使用输入的数据类型
        )
        
        # 恢复原始batch维度
        out_features = layer.weight.shape[0]  # 使用原始权重的输出维度
        output = output.view(*input_shape[:-1], out_features)
        
        # 添加bias，确保dtype一致
        if bias is not None:
            output = output + bias.to(original_dtype)
            
        return output

    def process_weights_after_loading(self, layer):
        """
        权重加载后的处理步骤
        
        Args:
            layer: 线性层模块
        """
        # 1. 打包int4权重到int32格式
        # 原始权重为int8保存的int4数据，需要打包为int32
        weight_packed = pack_int4_weights_with_npu(layer.weight.data)
        # 将打包后的权重注册为layer的缓冲区
        layer.register_buffer('weight_packed', weight_packed)
        
        # 转置权重（如果需要）不建议转置，npu_quant_matmul算子处理连续权重的逻辑和处理转置权重逻辑不同
        if self.transpose_weight:
            weight_packed_transposed = layer.weight_packed.transpose(0, 1).contiguous()
            layer.register_buffer('weight_packed', weight_packed_transposed)
        
        # 2. 确保weight_scale和weight_offset为float32
        layer.weight_scale.data = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.to(torch.float32)
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data,
                                                      ACL_FORMAT_FRACTAL_NZ)
        # 3 提前转置变换矩阵
        layer.left_trans = torch.nn.Parameter(layer.left_trans.data.t().contiguous())
        layer.right_trans = torch.nn.Parameter(layer.right_trans.data)
        # 4. 确保clip_ratio为float32标量

        layer.clip_ratio = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=layer.weight.device))
        # layer.clip_ratio = torch.nn.Parameter(layer.clip_ratio.data.to(torch.float32))

        layer.aclnn_clip_ratio = layer.clip_ratio.item()
        print(f"W4A4 FlatQuant Dynamic layer initialized: "
              f"original_weight_shape={layer.weight.shape}, "
              f"packed_weight_shape={layer.weight_packed.shape}, "
              f"transform_dims=({layer.left_trans.shape[0]}, {layer.right_trans.shape[0]}), "
              f"clip_ratio={layer.clip_ratio.item():.3f}") 


class AscendW4A4FlatQuantDynamicFakeLinearMethod:
    """Linear method for Ascend W4A4_FLATQUANT_DYNAMIC_FAKE.
    
    This class implements W4A4 fake quantization with FlatQuant approach using floating point simulation.
    - Weight: 4-bit quantization simulation (per-channel) with scale, stored as float
    - Activation: 4-bit dynamic quantization simulation with FlatQuant transform matrices
    - No complex preprocessing, packing, or NPU-specific operations required
    - Uses standard PyTorch operations for simulation
    """
    input_size = 0
    output_size = 0

    def __init__(self):
        self.transpose_weight = False
        self.sym = True  # 使用对称量化

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        """
        获取权重参数字典（伪量化版本）
        
        Args:
            input_size: 输入维度
            output_size: 输出维度
            params_dtype: 参数数据类型
            
        Returns:
            权重参数字典
        """
        AscendW4A4FlatQuantDynamicFakeLinearMethod.input_size = input_size
        AscendW4A4FlatQuantDynamicFakeLinearMethod.output_size = output_size
        params_dict = {
            # 原始浮点权重，用于伪量化模拟
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        """
        获取per-tensor量化参数字典（伪量化版本）
        
        Args:
            params_dtype: 参数数据类型
            
        Returns:
            per-tensor参数字典
        """
        params_dict = {}
        # FlatQuant变换矩阵（左变换和右变换）
        left_trans_dim, right_trans_dim = get_decompose_dim(AscendW4A4FlatQuantDynamicFakeLinearMethod.input_size)
        params_dict["left_trans"] = torch.empty(left_trans_dim, left_trans_dim, dtype=params_dtype)
        params_dict["right_trans"] = torch.empty(right_trans_dim, right_trans_dim, dtype=params_dtype)
        
        # 量化截断比例参数
        params_dict["clip_ratio"] = torch.empty(1, dtype=torch.float32)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """
        获取per-channel量化参数字典（伪量化版本）
        
        Args:
            output_size: 输出维度
            params_dtype: 参数数据类型
            
        Returns:
            per-channel参数字典
        """
        params_dict = {}
        # 权重量化scale (per-channel)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        # 权重量化offset (per-channel)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=torch.float32)
        return params_dict

    @staticmethod
    def get_qmin_qmax(bits, sym):
        """获取量化范围"""
        if sym:
            q_max = torch.tensor(2 ** (bits - 1) - 1)  # int4: 7
            q_min = -q_max - 1  # int4: -8
        else:
            q_max, q_min = torch.tensor(2 ** bits - 1), 0
        return q_max, q_min

    @staticmethod
    def get_scale_zero(x, clip_ratio, sym=True):
        """
        获取动态量化的scale和zero_point参数（per-token）
        模拟量化参数计算
        """
        q_max, q_min = AscendW4A4FlatQuantDynamicFakeLinearMethod.get_qmin_qmax(4, sym)  # int4对称量化: [-8, 7]
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)

        # 应用clip_ratio,此处乘法为正确用法，但算子是做除法
        xmax = xmax * clip_ratio
        xmin = xmin * clip_ratio
        
        if sym:
            # 对称量化：使用绝对值最大值
            xmax = torch.maximum(torch.abs(xmin), xmax)
            scale = (xmax / q_max).clamp(min=1e-8)  # 避免除零
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            # 非对称量化
            scale = (xmax - xmin) / (q_max - q_min)
            zero = torch.round(-xmin / scale)
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)   
            zero = zero.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero

    @staticmethod
    def kronecker_matmul(x, hadL, hadR):
        """kronecker乘积矩阵乘法"""
        init_shape = x.shape
        x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
        x = torch.matmul(x, hadR)
        x = torch.matmul(hadL.T, x)
        return x.reshape(init_shape)

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        应用W4A4 FlatQuant动态伪量化的前向传播
        
        Args:
            layer: 线性层模块
            x: 输入张量
            bias: 偏置项
            tp_rank: 张量并行rank
            
        Returns:
            输出张量
        """
        original_dtype = x.dtype
        input_shape = x.shape
        in_features = input_shape[-1]
        
        # 获取FlatQuant变换矩阵维度
        M = layer.left_trans.shape[0]
        N = layer.right_trans.shape[0]
        
        # 确保 M * N == in_features
        assert M * N == in_features, f"FlatQuant transform matrices dimension mismatch: M({M}) * N({N}) != in_features({in_features})"
        
        # 确保变换矩阵类型与输入匹配
        left_trans_matched = layer.left_trans.to(original_dtype)
        right_trans_matched = layer.right_trans.to(original_dtype)
        
        # 1. 先进行kronecker乘积变换（模拟FlatQuant的前处理）
        x_transformed = AscendW4A4FlatQuantDynamicFakeLinearMethod.kronecker_matmul(
            x, left_trans_matched, right_trans_matched
        )
        
        # 2. 对变换后的数据进行动态量化模拟
        scale, zero = AscendW4A4FlatQuantDynamicFakeLinearMethod.get_scale_zero(
            x_transformed, layer.clip_ratio, sym=True
        )
        
        # 执行量化：q = round((x - zero) / scale)
        x_quantized = torch.round(x_transformed / scale + zero).clamp(-8, 7)  # int4范围
        
        # 3. 反量化激活
        x_dequant = (x_quantized - zero) * scale
        

        x_dequant = x_dequant.view(-1, M * N)
        
        # 权重反量化
        weight_dequant = layer.weight.float() * layer.weight_scale
        
        # 5. 浮点矩阵乘法，确保类型匹配
        bias_matched = bias.to(original_dtype) if bias is not None else None

        output = torch.nn.functional.linear(
            x_dequant.to(original_dtype), 
            weight_dequant.to(original_dtype), 
            bias_matched
        )
        # 恢复原始batch维度
        out_features = layer.weight.shape[0]  # 使用原始权重的输出维度
        output = output.view(*input_shape[:-1], out_features)

        return output

    def process_weights_after_loading(self, layer):
        """
        权重加载后的处理步骤（伪量化版本）
        
        Args:
            layer: 线性层模块
        """
        # 1. 确保weight_scale为float32
        layer.weight_scale.data = layer.weight_scale.data.to(torch.float32)
        
        layer.clip_ratio = torch.nn.Parameter(layer.clip_ratio.data.to(torch.float32))
        
        print(f"W4A4 FlatQuant Dynamic Fake layer initialized: "
              f"weight_shape={layer.weight.shape}, "
              f"transform_dims=({layer.left_trans.shape[0]}, {layer.right_trans.shape[0]}), "
              f"clip_ratio={layer.clip_ratio.item():.3f}") 