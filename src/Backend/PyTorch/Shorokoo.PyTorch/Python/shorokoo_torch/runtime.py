"""The runtime half of Shorokoo's PyTorch backend: values in and out of the .NET side, the device
a run is on, and loading a translated model.

Tensors are torch tensors. String tensors are numpy arrays of dtype object, since torch has no
string dtype; they always stay on the host. Sequences are Python lists; an absent optional is
None. Element types travel as ONNX TensorProto.DataType codes.
"""

import contextvars
import ctypes
import linecache

import numpy as np
import torch

STRING = 8

_DTYPES = {
    1: torch.float32,
    2: torch.uint8,
    3: torch.int8,
    4: torch.uint16,
    5: torch.int16,
    6: torch.int32,
    7: torch.int64,
    9: torch.bool,
    10: torch.float16,
    11: torch.float64,
    12: torch.uint32,
    13: torch.uint64,
    14: torch.complex64,
    15: torch.complex128,
    16: torch.bfloat16,
    17: torch.float8_e4m3fn,
    18: torch.float8_e4m3fnuz,
    19: torch.float8_e5m2,
    20: torch.float8_e5m2fnuz,
}
_CODES = {dtype: code for code, dtype in _DTYPES.items()}

KIND_TENSOR = 0
KIND_SEQUENCE = 1
KIND_NONE = 2


def torch_dtype(code):
    """The torch dtype of an ONNX element type code; string and 4-bit types have none."""
    try:
        return _DTYPES[int(code)]
    except KeyError:
        raise NotImplementedError(f"ONNX element type {code} has no torch dtype") from None


def dtype_code(value):
    """The ONNX element type code of a tensor or a string tensor."""
    if isinstance(value, np.ndarray):
        return STRING
    return _CODES[value.dtype]


def is_strings(value):
    return isinstance(value, np.ndarray)


_device = contextvars.ContextVar("shorokoo_device", default=torch.device("cpu"))


def device():
    """The device of the run in progress: where an operator that makes a tensor from nothing puts it."""
    return _device.get()


# ---- values in -------------------------------------------------------------------------------

def from_host(address, nbytes, code, shape, device_name):
    """A tensor holding a copy of `nbytes` bytes at host `address`, on `device_name`."""
    tensor = torch.empty(tuple(shape), dtype=torch_dtype(code))
    if nbytes:
        ctypes.memmove(tensor.data_ptr(), address, nbytes)
    if device_name != "cpu":
        tensor = tensor.to(device_name)
    return tensor


def empty(code, shape, device_name):
    """An uninitialized tensor, in the memory of `device_name`."""
    return torch.empty(tuple(shape), dtype=torch_dtype(code), device=device_name)


def strings(values, shape):
    array = np.empty(len(values), dtype=object)
    array[:] = list(values)
    return array.reshape(tuple(shape))


def string_list(array):
    return [str(item) for item in array.reshape(-1).tolist()]


def host_copy(tensor):
    """A contiguous host tensor with `tensor`'s contents, which may be `tensor` itself."""
    return tensor.detach().to("cpu").contiguous()


def sequence_element(sequence, index):
    """An element of a sequence as a value of its own: a copy, so that writing to it cannot reach
    into the sequence."""
    element = sequence[index]
    return element.copy() if is_strings(element) else element.clone()


def describe(value):
    """(kind, element type code, shape, is host, data address, byte count) of a value."""
    if value is None:
        return (KIND_NONE, 0, [], True, 0, 0)
    if isinstance(value, list):
        code = dtype_code(value[0]) if value else 0
        return (KIND_SEQUENCE, code, [len(value)], True, 0, 0)
    if is_strings(value):
        return (KIND_TENSOR, STRING, list(value.shape), True, 0, 0)
    return (
        KIND_TENSOR,
        _CODES[value.dtype],
        list(value.shape),
        value.device.type == "cpu",
        value.data_ptr(),
        value.numel() * value.element_size(),
    )


# ---- runs ------------------------------------------------------------------------------------

def _storage_of(value):
    if value.numel() == 0:
        return None
    return value.untyped_storage().data_ptr()


def storages(values):
    """The storages a list of values holds, which an output must not share."""
    found = set()
    for value in values:
        if isinstance(value, list):
            found |= storages(value)
        elif isinstance(value, torch.Tensor):
            storage = _storage_of(value)
            if storage is not None:
                found.add(storage)
    return found


def _to_run_device(value, run_device):
    if isinstance(value, list):
        return [_to_run_device(item, run_device) for item in value]
    if isinstance(value, torch.Tensor) and value.device != run_device:
        return value.to(run_device)
    return value


def _export(value, retain, taken, ids):
    """An output made safe to hand over: contiguous, on the host unless it is retained on the
    device, and in memory of its own -- never an input's, a constant's or another output's, since
    the caller owns what it is handed and may write to it."""
    if value is None:
        return None
    if isinstance(value, list):
        return [_export(item, False, taken, ids) for item in value]
    if is_strings(value):
        if id(value) in ids:
            value = value.copy()
        ids.add(id(value))
        return value
    value = value.detach()
    if not retain:
        value = value.to("cpu")
    value = value.contiguous()
    storage = _storage_of(value)
    if storage is not None and storage in taken:
        value = value.clone()
        storage = _storage_of(value)
    if storage is not None:
        taken.add(storage)
    return value


def run(main, args, wanted, retained, run_device, constant_storages, constant_ids):
    """Runs a translated model and exports the outputs at indices `wanted`, each retained on the
    device where `retained` says so. Returns (value, description) per output."""
    run_device = torch.device(run_device)
    token = _device.set(run_device)
    try:
        moved = [_to_run_device(arg, run_device) for arg in args]
        with torch.no_grad():
            outputs = main(*moved)
    finally:
        _device.reset(token)
    taken = set(constant_storages) | storages(args) | storages(moved)
    ids = set(constant_ids) | {id(arg) for arg in args}
    results = []
    for index, retain in zip(wanted, retained):
        value = _export(outputs[index], retain, taken, ids)
        results.append((value, describe(value)))
    return results


# ---- models ----------------------------------------------------------------------------------

_compiled = {}


def load_model(source, filename, constants):
    """The `main` function of a translated model, with `constants` bound. The compiled code is
    cached by `filename`, which names the model's hash; the constants are the session's own."""
    code = _compiled.get(filename)
    if code is None:
        code = compile(source, filename, "exec")
        _compiled[filename] = code
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"__name__": "shorokoo_model", "_C": constants}
    exec(code, namespace)
    return namespace["main"]


def constant_storages(constants):
    return list(storages(constants))


def constant_ids(constants):
    return [id(value) for value in constants if is_strings(value)]


def torch_version():
    return str(torch.__version__)


def cuda_device_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0
