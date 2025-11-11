import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
from typing import Union, Literal

import torch
from safetensors.torch import load_file, save_file

#from kernel import weight_dequant
def unpack_from_int32(
    value: torch.Tensor,
    num_bits: int,
    shape: torch.Size,
    packed_dim: Union[Literal[0], Literal[1]] = 1,
) -> torch.Tensor:
    """
    Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
    original bit range.

    Return tensors in int8

    :param value: tensor to upack
    :param num_bits: number of bits to unpack each data point into
    :param shape: shape to unpack into, used to remove padding
    :returns: unpacked int8 tensor
    """
    if value.dtype is not torch.int32:
        raise ValueError(
            f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
        )

    if num_bits > 8:
        raise ValueError("Unpacking is only supported for less than 8 bits")

    pack_factor = 32 // num_bits

    # unpack
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked = torch.zeros(
            (value.shape[0], value.shape[1] * pack_factor),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

        # remove padding
        original_row_size = int(shape[1])
        unpacked = unpacked[:, :original_row_size]
    else:
        unpacked = torch.zeros(
            (value.shape[0] * pack_factor, value.shape[1]),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

        # remove padding
        original_row_size = int(shape[0])
        unpacked = unpacked[:original_row_size, :]

    # bits are packed in unsigned format, reformat to signed
    # update the value range from unsigned to signed
    offset = pow(2, num_bits) // 2
    unpacked = (unpacked - offset).to(torch.int8)

    return unpacked

def weight_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor for per-group quantization.
    The scale tensor is applied along the second dimension (columns) with the specified group size.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape (M, N).
        scale (torch.Tensor): The scale tensor of shape (M, N // group_size).
        group_size (int, optional): The group size used for quantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after grouping.
    """

    # Get the original dimensions of weight
    M, N = weight.shape


    # Compute the effective group dimensions for scale
    scale_m, scale_n = scale.shape
    group_size = N // scale_n
    assert N % scale_n == 0, f"N ({N}) is not divisible by K ({scale_n})"
    assert scale_m == M, f"Mismatch in scale rows ({scale_m}) and weight rows ({M})."

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)

    # Expand scale to match the weight tensor's shape along the second dimension
    # Repeat each scale value group_size times along the column dimension
    scale_expanded = scale.repeat_interleave(group_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:, :N]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight

def main(int4_path, bf16_path):
    """
    Converts INT4 weights to BF16 and saves the converted weights.

    This function reads INT4 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the 
    model index file to reflect the changes.

    Args:
    int4_path (str): The path to the directory containing the INT4 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale tensor is missing for a weight.

    Notes:
    - The function assumes that the INT4 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(int4_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensor files
    loaded_files = {}
    int4_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(int4_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(int4_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith(".weight_scale") or weight_name.endswith(".weight_shape"):
                continue
            elif weight_name.endswith(".weight_packed") and weight.dtype == torch.int32:
                # INT4 weight processing
                # Extract base name by removing ".weight_packed" suffix
                base_name = weight_name[:-len(".weight_packed")]
                scale_name = f"{base_name}.weight_scale"
                shape_name = f"{base_name}.weight_shape"
                try:
                    # Get scale and shape from the correct file
                    scale = get_tensor(scale_name)
                    original_shape = get_tensor(shape_name)
                    int4_weight_names.append(weight_name)
                    # Unpack and dequantize the INT4 weight
                    unpacked_weight = unpack_from_int32(weight, num_bits=4, shape=original_shape, packed_dim=1)
                    new_state_dict[weight_name] = weight_dequant(unpacked_weight, scale)
                except KeyError:
                    print(f"Warning: Missing scale or shape tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            # torch.cuda.empty_cache()
    
    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in int4_weight_names:
        # Extract base name by removing ".weight_packed" suffix
        base_name = weight_name[:-len(".weight_packed")]
        scale_name = f"{base_name}.weight_scale"
        shape_name = f"{base_name}.weight_shape"
        if scale_name in weight_map:
            weight_map.pop(scale_name)
        if shape_name in weight_map:
            weight_map.pop(shape_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-int4-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_int4_hf_path, args.output_bf16_hf_path)

