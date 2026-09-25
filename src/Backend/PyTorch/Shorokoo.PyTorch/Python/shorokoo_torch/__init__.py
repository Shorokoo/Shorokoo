"""Shorokoo's PyTorch backend support package.

The .NET backend translates an ONNX model into Python source that calls the helpers here, one
module per operator family (ops_elementwise, ops_logic, ...). The helpers carry the ONNX semantics;
the translator only decides which helper a node calls and with what. `runtime` holds everything
that is not an operator: value exchange with .NET, the run's device, and model loading.
"""
