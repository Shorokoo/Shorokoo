"""ONNX image and geometry (Resize, GridSample, RoiAlign, NonMaxSuppression, AffineGrid, ...).

No operator of this family is implemented yet: each one is added here, as a function named
after the operator, together with its entry in the C# operator table's file for this family. Until
then a model using one is refused when its session is created, naming the operator.
"""

import torch  # noqa: F401

from . import runtime as _rt  # noqa: F401
