#!/usr/bin/env python3
"""PyTorch reference for ModuleForwardValueTests.LinearForwardMatchesReference.

Loads the seeded Linear weights + fixed input exported by the _GoldenReferenceGen
harness, runs nn.functional.linear (Shorokoo Linear semantics: y = x @ W^T + b), and
prints the output as C# float[] initializer text to paste as the LinearReference constant.

Usage:
    # 1) export the weights+input from the C# side:
    PILOT_DIR=/tmp/pilot dotnet test tests/Shorokoo.Tests/Shorokoo.Tests.csproj \
        --filter "FullyQualifiedName~_GoldenReferenceGen"
    # 2) generate the reference:
    python3 tests/pytorch-reference/linear.py /tmp/pilot/linear.safetensors
"""
import sys
import torch
import torch.nn.functional as F
from safetensors.numpy import load_file

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pilot/linear.safetensors"
d = load_file(path)

weight = torch.tensor([v for k, v in d.items() if v.ndim == 2 and k != "input"][0])
bias = torch.tensor([v for k, v in d.items() if v.ndim == 1][0])
x = torch.tensor(d["input"])

y = F.linear(x, weight, bias).flatten().tolist()
print(",\n".join(
    "        " + ", ".join(f"{v:.8f}f" for v in y[i:i + 5]) for i in range(0, len(y), 5)
))
