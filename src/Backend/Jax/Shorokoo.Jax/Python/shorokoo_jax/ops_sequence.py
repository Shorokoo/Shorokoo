"""ONNX sequence and optional operators: none, since JAX has no sequences or optionals and the
translation refuses every model that uses one. The module exists because a translated model imports
every family's."""
