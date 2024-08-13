from torch import nn, Tensor, Size
from torch.types import Number
import numpy as np
from typing import Tuple, List, Union

__all__ = ['FUNCTION_COST_MAPPING']

def cost_elemwise(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input1 = args[0]
    input1 = Tensor([input1]) if not isinstance(input1, Tensor) else input1
    input_mem = input1.numel() * input1.element_size()
   
    if len(args) > 1:
        input2 = args[1]
        input2 = Tensor([input2]) if not isinstance(input2, Tensor) else input2
        input_mem += input2.numel() * input2.element_size()

    # FLOPs: n
    # Memory: 2*n
    
    if not isinstance(result, Tensor):
        num_elements = 1
        dtype_size = Tensor([result]).element_size()
    else:
        num_elements = result.numel()
        dtype_size = result.element_size()
        
    flops = num_elements

    # Assuming both inputs have same dtype
    output_mem = num_elements * dtype_size
    mem = input_mem + output_mem

    return flops, mem


def cost_addmatmul(result: Tensor, *args, **kwargs):
    assert len(args) == 3, len(args)
    input, tensor_A, tensor_B = args
    
    flops, mem = cost_matmul(result, tensor_A, tensor_B)
    
    flops += input.numel() * _prod(input.shape)
    mem += input.element_size() * _prod(input.shape)
    
    return flops, mem
    
def cost_matmul(result: Tensor, *args, **kwargs):
    
    assert len(args) == 2, len(args)
    tensor_A, tensor_B = args
    
    assert isinstance(tensor_A, Tensor) and isinstance(tensor_B, Tensor)
    def get_reduce_dim_shape(_s: Size, is_first_mat: bool):
        return _s[0] if len(_s) == 1 else _s[-1 if is_first_mat else -2]

    reduce_dim_shape = get_reduce_dim_shape(tensor_A.shape, True)
    assert reduce_dim_shape == get_reduce_dim_shape(tensor_B.shape, False)
    flops = (2 * reduce_dim_shape - 1) * result.shape.numel()
    
    input_mem = _prod(tensor_A.shape) * tensor_A.element_size() + _prod(tensor_B.shape) * tensor_B.element_size()
    output_mem = _prod(result.shape) * result.element_size()
    mem = input_mem + output_mem

    return flops, mem

# For nn.modules.*
def cost_functional_1d(module: nn.modules.conv._ConvNd, input_shape: Size, result_shape: Size, dtype_size: int=4) -> Tuple[int, int]:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    flops = (2 * kernel_size.numel() * module.in_channels - int(module.bias is None) * module.groups) * result_shape.numel()
    
    input_mem = input_shape.numel() * dtype_size
    weight_mem = module.weight.numel() * dtype_size
    bias_mem = module.out_channels * dtype_size if module.bias is not None else 0
    output_mem = result_shape.numel() * dtype_size
    mem = input_mem + weight_mem + bias_mem + output_mem
    
    return flops, mem


def cost_avgpoolnd(module: nn.modules.pooling._AvgPoolNd, input_shape: Size, result_shape: Size, dtype_size: int = 4) -> int:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    flops = kernel_size.numel() * result_shape.numel()
    
    input_mem = np.prod(input_shape) * dtype_size
    output_mem = np.product(result_shape) * dtype_size
    mem = input_mem + output_mem
    
    return flops, mem    


def cost_adaptive_avgpoolnd(module: nn.modules.pooling._AdaptiveAvgPoolNd, input_shape: Size, result_shape: Size, dtype_size: int = 4) -> int:
    kernel_size = Size(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(input_shape[2:], result_shape[2:])
    )
    
    flops = kernel_size.numel() * result_shape.numel()
    
    input_mem = np.prod(input_shape) * dtype_size
    output_mem = np.prod(result_shape) * dtype_size
    mem = input_mem + output_mem
    
    return flops, mem

def cost_maxpoolnd(module: nn.modules.pooling._MaxPoolNd, input_shape: Size, result_shape: Size, dtype_size: int = 4) -> Tuple[int, int]:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    
    flops = (kernel_size.numel() - 1) * result_shape.numel()
    
    input_mem = np.prod(input_shape) * dtype_size
    output_mem = np.prod(result_shape) * dtype_size
    mem = input_mem + output_mem

    return flops, mem

def flops_adaptive_maxpoolnd(module: nn.modules.pooling._AdaptiveMaxPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(input_shape[2:], result_shape[2:])
    )
    return (kernel_size.numel() - 1) * result_shape.numel()


def cost_adaptive_maxpoolnd(module: nn.modules.pooling._AdaptiveMaxPoolNd, input_shape: Size, result_shape: Size, dtype_size: int = 4) -> Tuple[int, int]:
    kernel_size = Size(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(input_shape[2:], result_shape[2:])
    )
    flops = (kernel_size.numel() - 1) * result_shape.numel()
    
    input_mem = np.prod(input_shape) * dtype_size
    output_mem = np.prod(result_shape) * dtype_size
    mem = input_mem + output_mem

    return flops, mem


def cost_sum(result: Tensor, *args, **kwargs) -> int:
    self_obj = args[0]
    this_shape = self_obj.shape
    result_shape = result.shape
    
    dim = args[1]
    keep_dim = args[2] if len(args) > 3 else False
    input_elements = self_obj.numel()
    total_flops = None
    if dim is None:
        # If no dimension is specified, sum over all elements
        total_flops = input_elements - 1
    else:
        # If dimensions are specified, calculate the FLOPs for reduction
        if not isinstance(dim, (list, tuple)):
            dim = [dim]
            
        if keep_dim:
            reduction_elements = [this_shape[d] for d in dim]
        else:
            reduction_elements  = [this_shape[d] for d in dim if d < len(this_shape)]
            
        reduction_size = 1
        for size in reduction_elements:
            reduction_size *= size
            
        total_flops = input_elements - reduction_size

        input_mem  = input_elements * self_obj.element_size()
        output_mem = result.numel() * result.element_size()
        
        mem = input_mem + output_mem
        
    return total_flops, mem


def cost_linalg_norm(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor
    ord = args[1] if len(args) > 1 else 2  # Default ord value is 2
    dim = args[2] if len(args) > 2 else None  # Default dim is None (compute norm of entire tensor)
    keepdim = args[3] if len(args) > 3 else False  # Default keepdim is False
    
    # Number of elements in the input tensor
    num_elements = input_tensor.numel()
    
    # FLOPs calculation
    if dim is None:
        # Frobenius norm (default when dim is None)
        flops = 2 * num_elements # Square and sum all elements, then sqrt
    elif isinstance(dim, tuple) and len(dim) == 2:
        # Matrix norm
        if ord in [1, float('inf')]:
            flops = num_elements - 1 # Sum along one dimension
        elif ord == -1 or ord == float('-inf'):
            flops = num_elements - 1 # Sum along one dimension
        elif ord == 2 or ord == -2:
            # Spectral norm (largest singular value)
            # This is an upper bound, actual cost depends on implementation
            flops = 10 * num_elements  # Approximate cost of SVD
        else:
            # Element-wise operation for other ord values
            flops = 2 * num_elements
    else:
        # Vector norm
        if ord == 0:
            flops = num_elements  # Count non-zero elements
        elif ord == 1:
            flops = num_elements - 1  # Sum of absolute values
        elif ord == 2:
            flops = 2 * num_elements  # Square, sum, then sqrt
        elif ord == float('inf') or ord == float('-inf'):
            flops = num_elements  # Find max or min absolute value
        else:
            flops = 3 * num_elements  # Power, sum, then root
    
    # Memory calculation
    input_dtype_size = input_tensor.element_size()
    output_dtype_size = result.element_size()
    
    # Read the input tensor
    input_mem = num_elements * input_dtype_size
    
    # Write the output tensor
    output_mem = result.numel() * output_dtype_size
    
    mem = input_mem + output_mem

    return flops, mem


def _prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

def cost_linalg_vector_norm(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor
    ord = args[1] if len(args) > 1 else 2  # Default ord value is 2
    dim = args[2] if len(args) > 2 else None  # Default dim is None (compute norm of entire tensor)
    keepdim = args[3] if len(args) > 3 else False  # Default keepdim is False
    
    # Number of elements in the input tensor
    num_elements = input_tensor.numel()
    
    # Determine the number of vectors
    if dim is None:
        num_vectors = 1
        vector_length = num_elements
    else:
        if isinstance(dim, int):
            dim = (dim,)
        num_vectors = num_elements // _prod(input_tensor.shape[d] for d in dim)
        vector_length = _prod(input_tensor.shape[d] for d in dim)
    
    # FLOPs calculation
    if ord == 0:
        flops = num_elements  # Count non-zero elements
    elif ord == 1:
        flops = num_elements  # Sum of absolute values
    elif ord == 2:
        flops = 2 * num_elements  # Square, sum, then sqrt for each vector
    elif ord == float('inf') or ord == float('-inf'):
        flops = num_elements  # Find max or min absolute value
    else:
        flops = 3 * num_elements  # Power, sum, then root for each vector
    
    # Add reduction operations
    flops += num_vectors  # One reduction (e.g., sqrt) per vector
    
    # Memory calculation
    input_dtype_size = input_tensor.element_size()
    output_dtype_size = result.element_size()
    
    # Read the input tensor
    input_mem = num_elements * input_dtype_size
    
    # Write the output tensor
    output_mem = result.numel() * output_dtype_size
    
    # Estimate intermediate memory (e.g., for temporary calculations)
    intermediate_mem = vector_length * input_dtype_size
    
    mem = input_mem + output_mem + intermediate_mem

    return flops, mem
    
def FunctionFLOPs_normalize(result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1, len(args)
    input_tensor = args[0]
    assert isinstance(input_tensor, Tensor)

    p = kwargs.get('p', 2)
    dim = kwargs.get('dim', 1)

    input_shape = input_tensor.shape
    n = input_shape[0]  # number of samples
    d = input_shape[1] if len(input_shape) > 1 else 1  # dimensionality of each sample
    
    # FLOPs for sum of squares, square root, and division
    total_flops = 2 * n * d + 1
    return total_flops

def cost_normalize(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor
    p = kwargs.get('p', 2)  # Default p value is 2
    dim = kwargs.get('dim', 1)  # Default dim is 1
    
    # Number of elements in the input tensor
    num_elements = input_tensor.numel()
    
    # Determine the number of normalization operations
    if isinstance(dim, int):
        dim = (dim,)
    num_normalizations = num_elements // _prod(input_tensor.shape[d] for d in dim)
    elements_per_normalization = _prod(input_tensor.shape[d] for d in dim)
    
    # FLOPs calculation
    # 1. Compute norm
    if p == 1:
        norm_flops = elements_per_normalization  # Sum of absolute values
    elif p == 2:
        norm_flops = 2 * elements_per_normalization  # Square, sum, then sqrt
    else:
        norm_flops = 3 * elements_per_normalization  # Power, sum, then root
    
    div_flops = num_elements # div
    
    total_flops = (norm_flops * num_normalizations) + div_flops
    
    # Memory calculation
    dtype_size = input_tensor.element_size()
    
    # Read the input tensor
    input_mem = num_elements * dtype_size
    
    # Write the output tensor
    output_mem = num_elements * dtype_size

    mem = input_mem + output_mem

    return total_flops, mem


def cost_tensordot(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    a = args[0]
    b = args[1]
    if len(kwargs) == 0:
        reduce_dims_a = tuple([a.shape[arg] for arg in args[2]])
        reduce_dims_b = tuple([b.shape[arg] for arg in args[3]])
    else:
        dims = kwargs['dims']
        assert len(dims[0]) == len(dims[1]) == 1
        reduce_dims_a = tuple([a.shape[arg] for arg in dims[0]])
        reduce_dims_b = tuple([a.shape[arg] for arg in dims[1]])
    assert reduce_dims_a == reduce_dims_b
    from functools import reduce
    import operator
    reduce_dim_shape = reduce(operator.mul, reduce_dims_b, 1)
    
    # FLOPs calculation
    flops = (2 * reduce_dim_shape - 1) * result.numel()
    
    # Memory calculation
    dtype_size = a.element_size()  # Assuming all tensors have the same data type
    
    # Read input tensors
    input_mem = a.numel() * dtype_size + b.numel() * dtype_size
    
    # Write output tensor
    output_mem = result.numel() * dtype_size
    
    # Total memory usage
    mem = input_mem + output_mem

    return flops, mem

def cost_einsum(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    from opt_einsum import contract_path
    import torch

    equation = args[0]
    operands = args[1:]
    if isinstance(operands[0], torch.fx.immutable_collections.immutable_list):
        operands = tuple(operands[0])
    
    # Calculate FLOPs using opt_einsum
    _, info = contract_path(equation, *operands)
    flops = int(info.opt_cost)
    
    # Memory calculation
    dtype_size = operands[0].element_size()  # Assuming all tensors have the same data type
    
    # Read input tensors
    input_mem = sum(operand.numel() * dtype_size for operand in operands)
    
    # Write output tensor
    output_mem = result.numel() * dtype_size
    
    # Estimate intermediate memory
    # This is a rough estimate based on the largest intermediate result
    largest_intermediate = max(info.size_dict.values(), default=0)
    intermediate_mem = largest_intermediate * dtype_size
    
    # Total memory usage
    mem = input_mem + output_mem + intermediate_mem

    return flops, mem

def FunctionFLOPs_linear(result: Tensor, *args, **kwargs) -> int:
    if len(args) == 3:
        input, weight, bias = args
    elif len(args) == 2:
        input, weight = args
        bias = kwargs.get('bias')
    else:
        input = args[0]
        weight = kwargs.get('weight')
        bias = kwargs.get('bias')

    assert isinstance(input, Tensor) and isinstance(weight, Tensor)

    total_flops = flops_matmul(input.shape, weight.T.shape, result.shape)
    if bias is not None:
        total_flops += flops_elemwise(result.shape)
    return total_flops 


def FunctionFLOPs_convnd(result: Tensor, *args, **kwargs) -> int:
    
    input = args[0]
    if len(args) > 1:
        weight = args[1]
    else:
        weight = kwargs.get('weight')

    assert isinstance(input, Tensor)
    assert isinstance(weight, Tensor)

    kernel_size = weight.shape[2:]
    in_channels = weight.shape[1]
    bias = kwargs.get('bias')
    groups = kwargs.get('groups', None)
    if groups is None:
        groups = 1
    stride = kwargs.get('stride', None)
    if stride is None:
        stride = 1
    padding = kwargs.get('padding', None)
    if padding is None:
        padding = 0
    result_shape = result.shape

    return flops_functional_convnd(bias, groups, kernel_size, in_channels, result_shape)

def cost_leaky_relu(result: Tensor, *args, **kwargs) -> int:
    return result.numel() * 4

def cost_interpolate(result: Tensor, *args, **kwargs) -> int:
    input = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size = kwargs.get('size', None)

    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            prod = 1
            for s in size:
                prod *= s
            return int(prod)
        else:
            return int(size)
    
    if len(args) > 2:
        scale_factor = args[2]
    else:
        scale_factor = kwargs.get('scale_factor', None)

    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        prod = 1
        for s in scale_factor:
            prod *= s
        flops *= int(prod)
    else:
        flops *= scale_factor**len(input)

    input_mem  = input.numel() * input.element_size()
    output_mem = result.numel() * result.element_size()
    
    mem = input_mem + output_mem

    return flops, mem

def cost_zero(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    return 0, 0

def cost_slice_tensor(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor

    flops = 0

    input_mem = input_tensor.numel() * input_tensor.element_size()
    output_mem = result.numel() * result.element_size()
    
    mem = input_mem + output_mem

    return flops, mem

def cost_slice_scatter(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor
    src = args[1]  # The source tensor containing values to scatter
    dim = args[2]  # The dimension along which to scatter
    start = args[3]  # The starting index for scattering
    end = args[4] if len(args) > 4 else input_tensor.size(dim)  # The end index for scattering
    step = args[5] if len(args) > 5 else 1  # The step size for scattering
    
    # Calculate the number of elements in the slice
    slice_size = (end - start + step - 1) // step
    num_elements_slice = slice_size * (src.numel() // src.size(dim))
    
    # Number of elements in the input tensor
    num_elements_input = input_tensor.numel()

    flops = 0
    
    # Memory calculation
    input_dtype_size = input_tensor.element_size()  # Get the size of the input data type in bytes
    src_dtype_size = src.element_size()  # Get the size of the source data type in bytes
    
    # Read the entire input tensor and the source tensor
    input_mem = num_elements_input * input_dtype_size
    src_mem = src.numel() * src_dtype_size
    
    # Write the updated slice to the output tensor
    output_mem = num_elements_slice * input_dtype_size
    
    mem = input_mem + src_mem + output_mem

    return flops, mem

def cost_cat(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    tensors = args[0]  # Assuming the first argument is a sequence of tensors
    dim = args[1] if len(args) > 1 else 0  # The dimension along which to concatenate
    
    # Total number of elements in all input tensors
    num_elements_input = sum(tensor.numel() for tensor in tensors)
    
    # Number of elements in the output tensor (same as total input elements)
    num_elements_output = num_elements_input

    flops = 0
    
    # Memory calculation
    input_mem = 0
    for tensor in tensors:
        input_mem += tensor.numel() * tensor.element_size()
    
    output_dtype_size = result.element_size()  # Get the size of the output data type in bytes
    
    # Read all input tensors
    # Write the output tensor
    output_mem = num_elements_output * output_dtype_size
    
    mem = input_mem + output_mem

    return flops, mem

def cost_broadcast(result: List[Tensor], *args, **kwargs) -> Tuple[int, int]:
    tensors = args[0]  # Assuming the first argument is a sequence of tensors

    flops = 0

    # Memory calculation
    input_mem = 0
    for tensor in tensors:
        input_mem += tensor.numel() * tensor.element_size()

    mem = input_mem

    return flops, mem

def cost_new(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    # Read the shape of the input tensor
    output_mem = result.numel() * result.element_size()

    flops = 0

    mem = output_mem

    return flops, mem

def cost_like(result: Tensor, *args, **kwargs) -> Tuple[int, int]:

    if args[0] is not None:
        input_tensor = args[0]  # Assuming the first argument is the input tensor
        # Read the shape of the input tensor (negligible, but we'll include it for completeness)
        input_mem = input_tensor.numel() * input_tensor.element_size()
    else:
        input_mem = 0

    flops = 0

    # Write the output tensor
    output_mem = result.numel()* result.element_size()
    
    mem = input_mem + output_mem

    return flops, mem

def cost_one_hot(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor
    num_classes = args[1]  # The number of classes for the one-hot encoding
    
    # Number of elements in the input tensor
    num_elements_input = input_tensor.numel()
    
    # Number of elements in the output tensor
    # This will be the number of input elements multiplied by the number of classes
    num_elements_output = num_elements_input * num_classes

    flops = 0
    
    # Memory calculation
    input_dtype_size = input_tensor.element_size()  # Get the size of the input data type in bytes
    output_dtype_size = result.element_size()  # Get the size of the output data type in bytes
    
    # Read the input tensor
    input_mem = num_elements_input * input_dtype_size
    
    # Write the one-hot encoded output tensor
    output_mem = num_elements_output * output_dtype_size
    
    mem = input_mem + output_mem

    return flops, mem

def cost_squeeze(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor
    dim = args[1] if len(args) > 1 else kwargs.get('dim', None)  # The dimension to squeeze
    
    # Number of elements in the input tensor
    # (which is the same as the number of elements in the output tensor)
    num_elements = input_tensor.numel()

    flops = 0
    
    # Memory calculation
    dtype_size = input_tensor.element_size()  # Get the size of the data type in bytes
    
    # Read the entire input tensor
    input_mem = num_elements * dtype_size
    
    # Write the entire output tensor
    # (same number of elements, but potentially different shape)
    output_mem = num_elements * dtype_size
    
    mem = input_mem + output_mem

    return flops, mem

def cost_select_int(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor
    dim = args[1]  # The dimension along which to select
    index = args[2]  # The index of the slice to select
    
    # Calculate the number of elements in the output tensor
    output_size = list(input_tensor.size())
    output_size.pop(dim)  # Remove the dimension we're selecting from
    num_elements_output = 1
    for s in output_size:
        num_elements_output *= s
    
    # Number of elements in the input tensor
    num_elements_input = input_tensor.numel()

    flops = 0
    
    # Memory calculation
    dtype_size = input_tensor.element_size()  # Get the size of the data type in bytes
    
    # Read the entire input tensor
    input_mem = num_elements_input * dtype_size
    
    # Write the selected elements to the output
    output_mem = num_elements_output * dtype_size
    
    mem = input_mem + output_mem

    return flops, mem

def cost_index_select(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor
    dim = args[1]  # The dimension along which to index
    index = args[2]  # The indices of elements to select
    
    # Number of elements in the index tensor
    num_indices = index.numel()
    
    # Number of elements in the input tensor
    num_elements_input = input_tensor.numel()
    
    # Number of elements in the output tensor
    # This will be the number of indices multiplied by the size of other dimensions
    num_elements_output = num_indices * (num_elements_input // input_tensor.size(dim))
    
    # FLOPs calculation
    # We count one operation per selected element for the selection process
    flops = 0
    
    # Memory calculation
    dtype_size = input_tensor.element_size()  # Get the size of the data type in bytes
    index_dtype_size = index.element_size()  # Get the size of the index data type
    
    # Read the entire input tensor and the index tensor
    input_mem = num_elements_input * dtype_size + num_indices * index_dtype_size
    
    # Write the selected elements to the output
    output_mem = num_elements_output * dtype_size
    
    mem = input_mem + output_mem

    return flops, mem

def cost_index_tensor(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    input_tensor = args[0]  # Assuming the first argument is the input tensor
    index_tensor = list(filter(lambda x: x is not None, args[1]))[0]  # Assuming the second argument is an index list and has one non-None element
    
    # Number of elements in the index tensor
    num_indices = index_tensor.numel()
    
    # Number of elements in the input tensor
    num_elements_input = input_tensor.numel()

    flops = 0

    # Read the entire input tensor and the index tensor
    input_mem = num_elements_input * input_tensor.element_size() + num_indices * index_tensor.element_size()
    
    # Write the selected elements to the output
    output_mem = num_indices * input_tensor.element_size()
    
    mem = input_mem + output_mem

    return flops, mem

def cost_silu(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    self_obj = args[0]
    
    # Number of elements in the input tensor
    num_elements = self_obj.numel()
    
    # FLOPs for sigmoid calculation per element:
    # sigmoid(x) = 1 / (1 + exp(-x))
    # This involves 1 exponential, 1 addition, and 1 division per element
    flops_per_sigmoid = 4  # considering exp, addition, division, and multiplication for 1 / (1 + exp(-x))
    
    # FLOPs for the multiplication with the original element
    flops_per_silu = flops_per_sigmoid + 1
    
    # Total FLOPs
    flops = num_elements * flops_per_silu

    # Memory calculation
    dtype_size = self_obj.element_size()  # Get the size of the data type in bytes
    input_mem = num_elements * dtype_size  # Read input tensor
    output_mem = num_elements * dtype_size  # Write output tensor
    mem = input_mem + output_mem

    return flops, mem

def cost_silu(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    self_obj = args[0]
    
    # Number of elements in the input tensor
    num_elements = self_obj.numel()
    
    # FLOPs for sigmoid calculation per element:
    # sigmoid(x) = 1 / (1 + exp(-x))
    # This involves 1 exponential, 1 addition, and 1 division per element
    flops_per_sigmoid = 4  # considering exp, addition, division, and multiplication for 1 / (1 + exp(-x))
        
    # Total FLOPs
    flops = num_elements * flops_per_sigmoid

    # Memory calculation
    dtype_size = self_obj.element_size()  # Get the size of the data type in bytes
    input_mem = num_elements * dtype_size  # Read input tensor
    output_mem = num_elements * dtype_size  # Write output tensor
    mem = input_mem + output_mem

    return flops, mem

def cost_scatter_add(result: Tensor, *args, **kwargs) -> Tuple[int, int]:
    self_obj = args[0]
    dim = args[1]
    index = args[2]
    src = args[3]

    # Ensure the index tensor and the source tensor have the same shape
    assert index.shape == src.shape, "Index and source tensors must have the same shape"

    # Number of elements in the index/source tensor
    n = index.numel()

    # Number of elements in the result tensor
    m = result.numel()

    # FLOPs: n (where n is the number of elements in index/src tensor)
    # Memory: 3n + m (where m is the number of elements in the result tensor)
    
    # FLOPs calculation
    flops = n  # One addition per element in index/src tensor

    # Memory calculation
    dtype_size = result.element_size()  # Get the size of the data type in bytes
    input_mem = (m + n + n) * dtype_size  # Read result, index, and src tensors
    output_mem = n * dtype_size  # Write to result tensor (only scattered elements)
    mem = input_mem + output_mem

    return flops, mem


FUNCTION_COST_MAPPING = {
    'sym_size': cost_zero,
    'transpose.int': cost_zero,
    't.default': cost_zero,
    'detach.default': cost_like,
    'empty.memory_format': cost_zero,
    'scatter_add.default': cost_scatter_add,
    'scatter_add_.default': cost_scatter_add,
    'silu.default': cost_silu,
    'index.Tensor': cost_index_tensor,
    'slice.Tensor': cost_like,
    'unsqueeze.default': cost_squeeze,
    'squeeze.dim': cost_squeeze,
    'index_select.default': cost_index_select,
    'select.int': cost_select_int,
    'one_hot.default': cost_one_hot,
    'ones_like.default': cost_like,
    'ones_like': cost_like,
    'zeros.default': cost_new,
    'new_zeros.default': cost_like,
    'scatter_.value': cost_like,
    'slice_scatter.default': cost_slice_scatter,
    '_to_copy.default': cost_like,
    'copy.default': cost_like,
    'copy_.default': cost_like,
    'concat': cost_cat,
    'cat': cost_cat,
    'cat.default': cost_cat,
    'clone.default': cost_like,
    'stack.default': cost_cat,
    'stack': cost_cat,
    '_unsafe_view.default': cost_zero,
    'view.default': cost_zero,
    'permute.default': cost_zero,
    'tensordot.default': cost_tensordot,
    'tensordot': cost_tensordot,
    'broadcast_tensors.default': cost_broadcast,
    'linalg_norm.default': cost_linalg_norm,
    'linalg_vector_norm.default': cost_linalg_vector_norm,
    'normalize': cost_normalize,
    'getattr': cost_zero,
    'getitem': cost_zero,
    'clamp_min.default': cost_elemwise,
    'einsum.default': cost_einsum,
    'einsum': cost_einsum,
    'mul.Scalar': cost_elemwise,
    'lt.Scalar': cost_elemwise,
    'cos.default': cost_elemwise,
    'sin.default': cost_elemwise,
    'tanh.default': cost_elemwise,
    'expand.default': cost_like,
    'pow.Tensor_Scalar': cost_elemwise,
    'div.Tensor': cost_elemwise,
    'add.Tensor': cost_elemwise,
    'rsub.Scalar': cost_elemwise,
    'sub.Scalar': cost_elemwise,
    'sub.Tensor': cost_elemwise,
    'mul.Tensor': cost_elemwise,
    'mul': cost_elemwise,
    'truediv': cost_elemwise,
    'sub': cost_elemwise,
    'addmm.default': cost_addmatmul,
    'bmm.default': cost_matmul,
    'mm.default': cost_matmul,
    'matmul.default': cost_matmul,
    'matmul': cost_matmul,
    'sum.dim_IntList': cost_sum,
    'add': cost_elemwise,
    '_assert': cost_zero,
    'eq': cost_elemwise,
    'rsqrt.default': cost_elemwise,
    'sigmoid.default': cost_elemwise,
    # 'linear.default': FunctionFLOPs_linear,
    # 'linear': FunctionFLOPs_linear,
    # 'conv1d': FunctionFLOPs_convnd,
    # 'conv2d': FunctionFLOPs_convnd,
    # 'conv3d': FunctionFLOPs_convnd,
    # 'leaky_relu': FunctionFLOPs_leaky_relu,
    # 'pad': FunctionFLOPs_zero,
    # 'floordiv': FunctionFLOPs_zero,
    # 'flip': FunctionFLOPs_zero,
    'interpolate': cost_interpolate,
}

# METHOD_COST_MAPPING = {
#     'narrow': MethodFLOPs_zero,
#     '__setitem__': MethodFLOPs_zero,
#     'reshape': MethodFLOPs_zero,
#     'permute': MethodFLOPs_zero,
#     'unbind': MethodFLOPs_zero,
#     'transpose': MethodFLOPs_zero,
#     'repeat': MethodFLOPs_zero,
#     'unsqueeze': MethodFLOPs_zero,
#     'exp': MethodFLOPs_elemwise,
#     'sum': MethodFLOPs_sum,
#     'div': MethodFLOPs_elemwise,
#     'softmax': MethodFLOPs_softmax,
#     'expand': MethodFLOPs_zero,
#     'flatten': MethodFLOPs_zero,
#     'view': MethodFLOPs_zero,
#     'cuda': MethodFLOPs_zero,
#     'flip': MethodFLOPs_zero,
#     'type_as': MethodFLOPs_zero,
#     'size': MethodFLOPs_zero,
#     'clone': MethodFLOPs_zero,
#     'new_empty': MethodFLOPs_zero,
#     'normal_': MethodFLOPs_zero,
#     'add_': MethodFLOPs_elemwise,
#     'pow': MethodFLOPs_zero,
# }
