# flake8: noqa

import torch
import e3nn
from e3nn import o3, nn
e3nn.set_optimization_defaults(jit_script_fx=False)

device=torch.device('cuda')
layer = 'sph'

if layer == 'linear':
    irreps = o3.Irreps("0e + 1o")
    x = irreps.randn(-1).to(device=device)
    model = o3.Linear(irreps, irreps).to(device=device)
    args = (x,)

if layer == 'sph':
    irreps =  o3.Irreps("0e + 1o")
    x = torch.randn(3).to(device=device)
    model = o3.SphericalHarmonics(irreps, normalize=True, normalization="component")
    model_symbol = torch.fx.symbolic_trace(model)
    args = (x,)

if layer == 'full_tp':
    irreps =  o3.Irreps("0e + 1o")
    x = irreps.randn(-1).to(device=device)
    model = o3.FullTensorProduct(irreps, irreps)
    model_symbol = torch.fx.symbolic_trace(model)
    args = (x,x)
    
if layer == 'gate':
    irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated = (
        o3.Irreps("16x0o"),
        [torch.tanh],
        o3.Irreps("32x0o"),
        [torch.tanh],
        o3.Irreps("16x1e+16x1o"),
    )
    model = nn.Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
    model_symbol = torch.fx.symbolic_trace(model)
    
import torch
from torch import nn, Tensor
from torch_flops.flops_engine import TorchFLOPsByFX

flops_counter = TorchFLOPsByFX(model) # Currently harcoding cuda
flops_counter.propagate(*args)

result_table = flops_counter.print_result_table()
# Print FLOPs, execution time and max GPU memory.
total_flops = flops_counter.print_total_flops(show=True)
# total_time = flops_counter.print_total_time()
# max_memory = flops_counter.print_max_memory()

