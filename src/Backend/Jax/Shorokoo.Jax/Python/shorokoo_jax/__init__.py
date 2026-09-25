"""Shorokoo's JAX backend support package.

The .NET backend translates an ONNX model into Python source that calls the helpers here, one
module per operator family (ops_elementwise, ops_logic, ...), under the same names and signatures
as every other Python-based backend's support package. The helpers carry the ONNX semantics in
jax.numpy and jax.lax; `runtime` holds everything that is not an operator: value exchange with
.NET, compiling a translated model with XLA per input-shape signature, and running it.

A translated model runs as one XLA program: its `main` is traced once per signature of input shapes
and element types and compiled. While it is traced, a value computed from shapes and constants
alone is *concrete* -- a numpy array -- and a value computed from an input is a tracer. The helpers
keep concrete what can be (see `runtime.xp`), which is what lets a shape computed in the graph
(Shape -> Gather -> Concat -> Reshape) reach an operator that needs it as a number; an operator
that needs a number where the graph computes it from an input's values cannot be compiled, and
says so (`runtime.DataDependentShape`).
"""
