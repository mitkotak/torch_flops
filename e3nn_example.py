# flake8: noqa

import torch
import e3nn
from e3nn import o3, nn
e3nn.set_optimization_defaults(jit_script_fx=False)

device=torch.device('cuda')
layer = 'full_tp'

if layer == 'mlp_linear':
    x = torch.randn(3)
    model = torch.nn.Linear(in_features=3, out_features=4)
    args = (x,) 
    
if layer == 'linear':
    irreps = o3.Irreps("1o")
    x = irreps.randn(-1).to(device=device)
    model = o3.Linear(irreps_in=irreps, irreps_out=irreps).to(device=device)
    args = (x, model.weight)

if layer == 'sph':
    irreps =  o3.Irreps("0e + 1o + 2e")
    x = torch.randn(3).to(device=device)
    model = o3.SphericalHarmonics(irreps, normalize=True, normalization="component")
    model_symbol = torch.fx.symbolic_trace(model)
    args = (x,)

if layer == 'full_tp':
    irreps =  o3.Irreps("0e + 1o + 2e")
    x = irreps.randn(-1)
    model = o3.FullTensorProduct(irreps, irreps)
    args = (x,x)

if layer == 'gate':
    irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated = (
        o3.Irreps("16x0o"),
        [torch.tanh],
        o3.Irreps("32x0o"),
        [torch.tanh],
        o3.Irreps("16x1e+16x1o"),
    )
    x = o3.Irreps("16x0o+32x0o+16x1e+16x1o").randn(-1)
    model = nn.Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
    args = (x,)
    model(x)

import torch
from torch import nn, Tensor
from torch_flops.flops_engine import TorchFLOPsByFX

flops_counter = TorchFLOPsByFX(model, args) # Currently harcoding cuda
flops_counter.propagate(*args)

result_table = flops_counter.print_result_table()
# Print FLOPs, execution time and max GPU memory.
for lmax in range(1, 9):
    irreps =  o3.Irreps.spherical_harmonics(lmax)
    x = irreps.randn(-1)
    model = o3.FullTensorProduct(irreps, irreps)
    args = (x,x)
    flops_counter = TorchFLOPsByFX(model, args) # Currently harcoding cuda
    flops_counter.propagate(*args)
    total_flops = flops_counter.print_total_flops(show=False)
    print(f"lmax {lmax}", total_flops)
# total_time = flops_counter.print_total_time()
# max_memory = flops_counter.print_max_memory()

