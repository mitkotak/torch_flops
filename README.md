# torch_flops

Based on [torch-flops](https://github.com/zugexiaodui/torch_flops/) with two caveats:

-  Swapped `symbolic_trace` with `make_fx`
-  Manually count up the bytes