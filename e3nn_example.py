# flake8: noqa

from e3nn.util.datatypes import Path, Chunk
from e3nn import o3

import torch
from torch import nn
import numpy as np


def _prepare_inputs(input1, input2):
    dtype = torch.promote_types(input1.dtype, input2.dtype)

    input1 = input1.to(dtype=dtype)
    input2 = input2.to(dtype=dtype)

    leading_shape = torch.broadcast_shapes(input1.shape[:-1], input2.shape[:-1])
    input1 = input1.broadcast_to(leading_shape + (-1,))
    input2 = input2.broadcast_to(leading_shape + (-1,))
    return input1, input2, leading_shape


class FullTensorProduct(nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        *,
        filter_ir_out: o3.Irreps = None,
        irrep_normalization: str = "component",
        regroup_output: bool = True,
    ):
        """Tensor Product adapted from https://github.com/e3nn/e3nn-jax/blob/cf37f3e95264b34587b3a202ea4c3eb82597307e/e3nn_jax/_src/tensor_products.py#L40-L135"""
        super(FullTensorProduct, self).__init__()

        if regroup_output:
            irreps_in1 = o3.Irreps(irreps_in1).regroup()
            irreps_in2 = o3.Irreps(irreps_in2).regroup()

        paths = {}
        irreps_out = []
        for (mul_1, ir_1), slice_1 in zip(irreps_in1, irreps_in1.slices()):
            for (mul_2, ir_2), slice_2 in zip(irreps_in2, irreps_in2.slices()):
                for ir_out in ir_1 * ir_2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue
                    cg = o3.wigner_3j(ir_1.l, ir_2.l, ir_out.l)
                    if irrep_normalization == "component":
                        cg *= ir_out.dim**0.5
                    elif irrep_normalization == "norm":
                        cg *= (ir_1.dim * ir_2.dim)**0.5
                    else:
                        raise ValueError(f"irrep_normalization={irrep_normalization} not supported")
                    chunk = torch.zeros((mul_1, mul_2, ir_out.dim))
                    self.register_buffer(f"chunk_{ir_1.l}_{ir_2.l}_{ir_out.l}", chunk)
                    self.register_buffer(f"cg_{ir_1.l}_{ir_2.l}_{ir_out.l}", cg)
                    paths[(ir_1.l, ir_1.p, ir_2.l, ir_2.p, ir_out.l, ir_out.p)] = Path(
                        Chunk(mul_1, ir_1.dim, slice_1), Chunk(mul_2, ir_2.dim, slice_2), Chunk(mul_1 * mul_2, ir_out.dim)
                    )
                    irreps_out.append((mul_1 * mul_2, ir_out))
        self.paths = paths
        irreps_out = o3.Irreps(irreps_out)
        self.irreps_out, _, self.inv = irreps_out.sort()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
    ) -> torch.Tensor:
        chunks = []
        leading_shape = ()
        for (l1, _, l2, _, l3, _), (
            (mul_1, input_dim1, slice_1),
            (mul_2, input_dim2, slice_2),
            (output_mul, output_dim, _),
        ) in self.paths.items():
            x1 = input1[..., slice_1].reshape(
                leading_shape
                + (
                    mul_1,
                    input_dim1,
                )
            )
            x2 = input2[..., slice_2].reshape(
                leading_shape
                + (
                    mul_2,
                    input_dim2,
                )
            )
            cg = getattr(self, f"cg_{l1}_{l2}_{l3}")
            # chunk = getattr(self, f"chunk_{l1}_{l2}_{l3}")
            # for u in range(mul_1):
            #     for v in range(mul_2):
            #       for i in range(input_dim1):
            #         for j in range(input_dim2):
            #           for k in range(output_dim):
            #             chunk[u, v, k] += x1[u,i] * x2[v,j] * cg[i,j,k]
            chunk = torch.einsum("...ui, ...vj, ijk -> ...uvk", x1, x2, cg)
            chunk = torch.reshape(chunk, leading_shape + (output_mul * output_dim,))
            chunks.append(chunk)

        return torch.cat([chunks[i] for i in self.inv], dim=-1)
    
import e3nn
from e3nn import o3
e3nn.set_optimization_defaults(jit_script_fx=False)

irreps = o3.Irreps("0e + 1o")
device=torch.device('cuda')

x = irreps.randn(-1).to(device=device)
model = o3.Linear(irreps, irreps).to(device=device)

import torch
from torch import nn, Tensor
from torch_flops.flops_engine import TorchFLOPsByFX

flops_counter = TorchFLOPsByFX(model) # Currently harcoding cuda
flops_counter.propagate(x,x)

result_table = flops_counter.print_result_table()
# Print FLOPs, execution time and max GPU memory.
total_flops = flops_counter.print_total_flops(show=True)
# total_time = flops_counter.print_total_time()
# max_memory = flops_counter.print_max_memory()

