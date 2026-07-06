#!/usr/bin/env python3
"""PyTorch reference for the Linear forward-value check (NNLinearMatchesManualMatMul).

Given a .safetensors file holding a Linear layer's seeded weight + bias and an
"input" tensor, runs nn.functional.linear (Shorokoo Linear semantics:
y = x @ W^T + b) and prints the flattened output as C# Vector(...) initializer
text to paste as the module's `reference`.

Usage:
    python3 tests/pytorch-reference/linear.py /path/to/linear.safetensors

The .safetensors is produced on the C# side by materializing the layer at the
test's MasterSeed, InitializeTrainableParams, and saving each param plus the
"input" tensor (see this directory's README).
"""
import sys
import torch
import torch.nn.functional as F
from safetensors.numpy import load_file

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/linear.safetensors"
d = load_file(path)

weight = torch.tensor([v for k, v in d.items() if v.ndim == 2 and k != "input"][0])
bias = torch.tensor([v for k, v in d.items() if v.ndim == 1][0])
x = torch.tensor(d["input"])

y = F.linear(x, weight, bias).flatten().tolist()
print("var reference = Vector(" + ", ".join(f"{v:.8f}f" for v in y) + ");")
