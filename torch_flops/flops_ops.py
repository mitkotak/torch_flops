from torch import nn, Tensor, Size
from torch.types import Number

__all__ = ['MODULE_FLOPs_MAPPING', 'FUNCTION_FLOPs_MAPPING', 'METHOD_FLOPs_MAPPING']


def flops_zero() -> int:
    return 0


def flops_elemwise(result_shape: Size) -> int:
    return result_shape.numel()


def flops_matmul(tensor1_shape: Size, tensor2_shape: Size, result_shape: Size) -> int:
    # 可根据输入维度改为分情况处理，参考https://github.com/zhijian-liu/torchprofile/blob/6d80fe57bb8c6bc9f789da7925fac6547fa9502b/torchprofile/handlers.py#L35
    def get_reduce_dim_shape(_s: Size, is_first_mat: bool):
        return _s[0] if len(_s) == 1 else _s[-1 if is_first_mat else -2]

    reduce_dim_shape = get_reduce_dim_shape(tensor1_shape, True)
    assert reduce_dim_shape == get_reduce_dim_shape(tensor2_shape, False)
    return (2 * reduce_dim_shape - 1) * result_shape.numel()

# For nn.modules.*
def flops_convnd(module: nn.modules.conv._ConvNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    return (2 * kernel_size.numel() * module.in_channels - int(module.bias is None) * module.groups) * result_shape.numel()


def flops_avgpoolnd(module: nn.modules.pooling._AvgPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    return kernel_size.numel() * result_shape.numel()


def flops_adaptive_avgpoolnd(module: nn.modules.pooling._AdaptiveAvgPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(input_shape[2:], result_shape[2:])
    )
    return kernel_size.numel() * result_shape.numel()


def flops_maxpoolnd(module: nn.modules.pooling._AvgPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    return (kernel_size.numel() - 1) * result_shape.numel()


def flops_adaptive_maxpoolnd(module: nn.modules.pooling._AdaptiveMaxPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(input_shape[2:], result_shape[2:])
    )
    return (kernel_size.numel() - 1) * result_shape.numel()


def flops_functional_convnd(bias: int, groups: int, kernel_size: Size, in_channels: int, result_shape: Size) -> int:
    total_flops = (2 * kernel_size.numel() * in_channels - int(bias is None) * groups) * result_shape.numel()
    return total_flops


# For ModuleFLOPs
def ModuleFLOPs_zero(module: nn.Linear, result: Tensor, *args, **kwargs) -> int:
    return flops_zero()


def ModuleFLOPs_elemwise(module: nn.Module, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape  # [..., d_in]
    result_shape = result.shape
    assert input_shape == result_shape

    total_flops = flops_elemwise(result_shape)
    return total_flops


def ModuleFLOPs_LeakyReLU(module: nn.LeakyReLU, result: Tensor, *args, **kwargs) -> int:
    return result.numel() * 4


def ModuleFLOPs_Linear(module: nn.Linear, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape  # [..., d_in]
    weight_shape = module.weight.T.shape  # [d_out, d_in].T -> [d_in, d_out]
    result_shape = result.shape

    assert input_shape[-1] == weight_shape[0], f"{input_shape}, {weight_shape}"
    matmul_shape = Size(list(input_shape[:-1]) + list(weight_shape[-1:]))
    assert matmul_shape == result_shape

    total_flops = flops_matmul(input_shape, weight_shape, result_shape)
    if module.bias is not None:
        total_flops += flops_elemwise(result_shape)

    return total_flops


def ModuleFLOPs_ConvNd(module: nn.Conv1d | nn.Conv2d | nn.Conv3d, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_convnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_AvgPoolNd(module: nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_avgpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_AdaptiveAvgPoolNd(module: nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_adaptive_avgpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_MaxPoolNd(module: nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_maxpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_AdaptiveMaxPoolNd(module: nn.AdaptiveMaxPool1d | nn.AdaptiveMaxPool2d | nn.AdaptiveMaxPool3d, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_adaptive_maxpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_Norm(module: nn.modules.batchnorm._NormBase | nn.LayerNorm | nn.GroupNorm, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)
    assert not module.training, "Only support `eval` mode."

    input_shape = args[0].shape  # [..., d_in]
    result_shape = result.shape
    assert input_shape == result_shape

    # (X-mean)/std
    total_flops = flops_elemwise(input_shape) * 2
    if (hasattr(module, 'affine') and module.affine) or (hasattr(module, 'elementwise_affine'), module.elementwise_affine):
        total_flops += flops_elemwise(input_shape) * 2

    return total_flops


def ModuleFLOPs_GELU(module: nn.GELU, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape  # [..., d_in]
    result_shape = result.shape
    assert input_shape == result_shape

    total_flops = flops_elemwise(result_shape)
    if module.approximate is None:
        raise NotImplementedError()

    return total_flops


# For FunctionFLOPs
def FunctionFLOPs_zero(result: Tensor, *args, **kwargs) -> int:
    return flops_zero()


def FunctionFLOPs_trig(result: Tensor | Number, *args, **kwargs) -> int:
    assert len(args) == 1, len(args)
    return flops_elemwise(result.shape)

def FunctionFLOPs_sum(result: Tensor, *args, **kwargs) -> int:
    self_obj = args[0]
    this_shape = self_obj.shape
    result_shape = result.shape
    
    dim = args[1]
    keep_dim = args[2]
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

    return total_flops

def FunctionFLOPs_elemwise(result: Tensor | Number, *args, **kwargs) -> int:
    assert len(args) == 2, len(args)

    total_flops = None
    if isinstance(result, Number):
        total_flops = 1
    elif isinstance(result, Tensor):
        total_flops = flops_elemwise(result.shape)
    elif isinstance(result, Size):
        total_flops = 0
    else:
        raise TypeError(type(result))

    return total_flops

def FunctionFLOPs_matmul(result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 2, len(args)
    tensor_A, tensor_B = args
    assert isinstance(tensor_A, Tensor) and isinstance(tensor_B, Tensor)

    total_flops = flops_matmul(tensor_A.shape, tensor_B.shape, result.shape)
    return total_flops


def FunctionFLOPS_linalg_norm(result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 3, len(args)
    input_tensor = args[0]
    p = args[1] if args[1] is not None else 2 # Default to 2-norm if not specified
    dim = args[2] 

    num_elements = input_tensor.numel()

    if isinstance(dim, (tuple, list)):
        dim = list(dim)
    else:
        dim = [dim] if dim is not None else []

    # Compute the number of elements along the dimensions to be reduced
    if dim:
        reduced_shape = list(input_tensor.shape)
        for d in sorted(dim, reverse=True):
            reduced_shape.pop(d)
        import torch
        num_elements_after_reduction = torch.prod(torch.tensor(reduced_shape)).item()
    else:
        num_elements_after_reduction = 1

    # Compute FLOPs based on the norm type
    if p == 1:  # L1 norm
        # Absolute value for each element + sum
        total_flops += 2 * num_elements - 1 # sum + # abs
        if dim:
            total_flops *= num_elements_after_reduction

    elif p == 2:  # L2 norm
        # Square each element + sum
        total_flops = 3*num_elements + 1 # square + sum + sqrt
        if dim:
            total_flops *= num_elements_after_reduction

    return total_flops

def FunctionFLOPS_linalg_vector_norm(result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 4, len(args)
    input_tensor = args[0]
    assert isinstance(input_tensor, Tensor)
    p = int(args[1])
    dim = args[2][0]
    assert isinstance(dim, int)

    num_elements = input_tensor.numel()

    if p == 1:  # L1 norm
        # Absolute value for each element + sum
       total_flops = num_elements + (num_elements - 1)
    elif p == 2:  # L2 norm
        # Square each element + sum + square root
        total_flops = num_elements + (num_elements - 1)
    elif p == float('inf'):  # Infinity norm
        # Absolute value for each element + comparisons for max
        total_flops = num_elements + (num_elements - 1)
    else:  # General case
        # Power operation + sum + nth root
        total_flops = num_elements * 2 + (num_elements - 1) + 1

    # If dim is specified, we're calculating multiple norms
    if dim is not None:
        # Calculate the number of norms we're computing
        num_norms = num_elements // input_tensor.size(dim)
        total_flops *= num_norms
    
    return total_flops
    
    
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

def FunctionFLOPs_tensordot(result: Tensor, *args, **kwargs) -> int:
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
    return (2 * reduce_dim_shape - 1) * result.shape.numel()

def FunctionFLOPs_einsum(result: Tensor, *args, **kwargs) -> int:
    from opt_einsum import contract_path
    import torch
    equation = args[0]
    operands = args[1:]
    if isinstance(operands[0], torch.fx.immutable_collections.immutable_list):
        operands = tuple(operands[0])
    _, info = contract_path(equation, *operands)
    total_flops = int(info.opt_cost)
    return total_flops

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

def FunctionFLOPs_leaky_relu(result: Tensor, *args, **kwargs) -> int:
    return result.numel() * 4

def FunctionFLOPs_interpolate(result: Tensor, *args, **kwargs) -> int:
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

    return flops

def FunctionFLOPs_silu(result: Tensor, *args, **kwargs) -> int:
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
    total_flops = num_elements * flops_per_silu
    
    return total_flops

def FunctionFLOPs_scatter_add(result: Tensor, *args, **kwargs) -> int:
    self_obj = args[0]
    dim = args[1]
    index = args[2]
    src = args[3]

    # Ensure the index tensor and the source tensor have the same shape
    assert index.shape == src.shape, "Index and source tensors must have the same shape"

    # Number of elements in the index/source tensor
    num_elements = index.numel()

    # Each scatter_add operation involves one addition per element
    total_flops = num_elements

    return total_flops

# For MethodFLOPs
def MethodFLOPs_zero(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    return flops_zero()


def MethodFLOPs_elemwise(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    return flops_elemwise(result.shape)


def MethodFLOPs_sum(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    this_shape = self_obj.squeeze().shape
    result_shape = result.squeeze().shape

    total_flops = None
    if len(result_shape) == 0:
        total_flops = self_obj.numel() - 1
    else:
        kept_shape = list(this_shape)
        for s in result_shape:
            kept_shape.remove(s)
        kept_shape = Size(kept_shape)
        total_flops = kept_shape.numel() * (result_shape.numel() - 1)

    return total_flops


def MethodFLOPs_softmax(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    this_shape = self_obj.shape
    result_shape = result.shape
    assert this_shape == result_shape

    exp_flops = flops_elemwise(this_shape)

    dim_reduce: int = args_tail[0] if args_tail else kwargs.get('dim')
    dims_kept = list(this_shape)
    dims_kept.pop(dim_reduce)
    dims_kept = Size(dims_kept)
    sum_flops = (this_shape[dim_reduce] - 1) * dims_kept.numel()

    div_flops = flops_elemwise(this_shape)

    total_flops = exp_flops + sum_flops + div_flops
    return total_flops
    


MODULE_FLOPs_MAPPING = {
    'Linear': ModuleFLOPs_Linear,
    'Identity': ModuleFLOPs_zero,
    'Conv1d': ModuleFLOPs_ConvNd,
    'Conv2d': ModuleFLOPs_ConvNd,
    'Conv3d': ModuleFLOPs_ConvNd,
    'AvgPool1d': ModuleFLOPs_AvgPoolNd,
    'AvgPool2d': ModuleFLOPs_AvgPoolNd,
    'AvgPool3d': ModuleFLOPs_AvgPoolNd,
    'AdaptiveAvgPool1d': ModuleFLOPs_AdaptiveAvgPoolNd,
    'AdaptiveAvgPool2d': ModuleFLOPs_AdaptiveAvgPoolNd,
    'AdaptiveAvgPool3d': ModuleFLOPs_AdaptiveAvgPoolNd,
    'MaxPool1d': ModuleFLOPs_MaxPoolNd,
    'MaxPool2d': ModuleFLOPs_MaxPoolNd,
    'MaxPool3d': ModuleFLOPs_MaxPoolNd,
    'AdaptiveMaxPool1d': ModuleFLOPs_AdaptiveMaxPoolNd,
    'AdaptiveMaxPool2d': ModuleFLOPs_AdaptiveMaxPoolNd,
    'AdaptiveMaxPool3d': ModuleFLOPs_AdaptiveMaxPoolNd,
    'LayerNorm': ModuleFLOPs_Norm,
    'BatchNorm1d': ModuleFLOPs_Norm,
    'BatchNorm2d': ModuleFLOPs_Norm,
    'BatchNorm3d': ModuleFLOPs_Norm,
    'InstanceNorm1d': ModuleFLOPs_Norm,
    'InstanceNorm2d': ModuleFLOPs_Norm,
    'InstanceNorm3d': ModuleFLOPs_Norm,
    'GroupNorm': ModuleFLOPs_Norm,
    'Dropout': ModuleFLOPs_zero,
    'GELU': ModuleFLOPs_GELU,
    'ReLU': ModuleFLOPs_elemwise,
    'Flatten': ModuleFLOPs_zero,
    'LeakyReLU': ModuleFLOPs_LeakyReLU,
    'type_as': ModuleFLOPs_zero
}
FUNCTION_FLOPs_MAPPING = {
    'scatter_add.default': FunctionFLOPs_scatter_add,
    'silu.default': FunctionFLOPs_silu,
    'index.Tensor': FunctionFLOPs_zero,
    'unsqueeze.default': FunctionFLOPs_zero,
    'squeeze.dim': FunctionFLOPs_zero,
    'index_select.default': FunctionFLOPs_zero,
    'select.int': FunctionFLOPs_zero,
    'one_hot.default': FunctionFLOPs_zero,
    'ones_like.default': FunctionFLOPs_zero,
    'new_zeros.default': FunctionFLOPs_zero,
    'slice_scatter.default': FunctionFLOPs_zero,
    '_to_copy.default': FunctionFLOPs_zero,
    'copy.default': FunctionFLOPs_zero,
    'slice.Tensor': FunctionFLOPs_zero,
    'cat.default': FunctionFLOPs_zero,
    'clone.default': FunctionFLOPs_zero,
    '_unsafe_view.default': FunctionFLOPs_zero,
    'view.default': FunctionFLOPs_zero,
    'permute.default': FunctionFLOPs_zero,
    'tensordot.default': FunctionFLOPs_tensordot,
    'tensordot': FunctionFLOPs_tensordot,
    'stack.default': FunctionFLOPs_zero,
    'stack': FunctionFLOPs_zero,
    'ones_like': FunctionFLOPs_zero,
    'broadcast_tensors.default': FunctionFLOPs_zero,
    'linalg_norm.default': FunctionFLOPS_linalg_norm,
    'linalg_vector_norm.default': FunctionFLOPS_linalg_vector_norm,
    'normalize': FunctionFLOPs_normalize,
    'getattr': FunctionFLOPs_zero,
    'getitem': FunctionFLOPs_zero,
    'clamp_min.default': FunctionFLOPs_elemwise,
    'einsum.default': FunctionFLOPs_einsum,
    'einsum': FunctionFLOPs_einsum,
    'lt.Scalar': FunctionFLOPs_elemwise,
    'sin.default': FunctionFLOPs_trig,
    'tanh.default': FunctionFLOPs_trig,
    'expand.default': FunctionFLOPs_zero,
    'pow.Tensor_Scalar': FunctionFLOPs_elemwise,
    'div.Tensor': FunctionFLOPs_elemwise,
    'add.Tensor': FunctionFLOPs_elemwise,
    'rsub.Scalar': FunctionFLOPs_elemwise,
    'sub.Scalar': FunctionFLOPs_elemwise,
    'sub.Tensor': FunctionFLOPs_elemwise,
    'mul.Tensor': FunctionFLOPs_elemwise,
    'mul': FunctionFLOPs_elemwise,
    'truediv': FunctionFLOPs_elemwise,
    'sub': FunctionFLOPs_elemwise,
    'matmul.default': FunctionFLOPs_matmul,
    'matmul': FunctionFLOPs_matmul,
    'sum.dim_IntList': FunctionFLOPs_sum,
    'add': FunctionFLOPs_elemwise,
    'concat': FunctionFLOPs_zero,
    '_assert': FunctionFLOPs_zero,
    'eq': FunctionFLOPs_elemwise,
    'cat': FunctionFLOPs_zero,
    'linear.default': FunctionFLOPs_linear,
    'linear': FunctionFLOPs_linear,
    'conv1d': FunctionFLOPs_convnd,
    'conv2d': FunctionFLOPs_convnd,
    'conv3d': FunctionFLOPs_convnd,
    'leaky_relu': FunctionFLOPs_leaky_relu,
    'pad': FunctionFLOPs_zero,
    'floordiv': FunctionFLOPs_zero,
    'flip': FunctionFLOPs_zero,
    'interpolate': FunctionFLOPs_interpolate,
}
METHOD_FLOPs_MAPPING = {
    'narrow': MethodFLOPs_zero,
    '__setitem__': MethodFLOPs_zero,
    'reshape': MethodFLOPs_zero,
    'permute': MethodFLOPs_zero,
    'unbind': MethodFLOPs_zero,
    'transpose': MethodFLOPs_zero,
    'repeat': MethodFLOPs_zero,
    'unsqueeze': MethodFLOPs_zero,
    'exp': MethodFLOPs_elemwise,
    'sum': MethodFLOPs_sum,
    'div': MethodFLOPs_elemwise,
    'softmax': MethodFLOPs_softmax,
    'expand': MethodFLOPs_zero,
    'flatten': MethodFLOPs_zero,
    'view': MethodFLOPs_zero,
    'cuda': MethodFLOPs_zero,
    'flip': MethodFLOPs_zero,
    'type_as': MethodFLOPs_zero,
    'size': MethodFLOPs_zero,
    'clone': MethodFLOPs_zero,
    'new_empty': MethodFLOPs_zero,
    'normal_': MethodFLOPs_zero,
    'add_': MethodFLOPs_elemwise,
    'pow': MethodFLOPs_zero,
}
