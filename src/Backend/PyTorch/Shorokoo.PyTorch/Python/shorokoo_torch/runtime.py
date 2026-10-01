"""The runtime half of Shorokoo's PyTorch backend: values in and out of the .NET side, the device
a run is on, and loading a translated model.

Tensors are torch tensors. String tensors are numpy arrays of dtype object, since torch has no
string dtype; they always stay on the host. Sequences are Python lists; an absent optional is
None. Element types travel as ONNX TensorProto.DataType codes.
"""

import collections
import contextvars
import ctypes
import linecache
import warnings

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
    return element.copy() if is_strings(element) else element.clone(memory_format=torch.contiguous_format)


def describe(value):
    """(kind, element type code, shape, is host, data address, byte count, CUDA device index) of a
    value; the device index is -1 for a value in host memory."""
    if value is None:
        return (KIND_NONE, 0, [], True, 0, 0, -1)
    if isinstance(value, list):
        code = dtype_code(value[0]) if value else 0
        return (KIND_SEQUENCE, code, [len(value)], True, 0, 0, -1)
    if is_strings(value):
        return (KIND_TENSOR, STRING, list(value.shape), True, 0, 0, -1)
    if not value.is_contiguous():
        # The .NET side reads a host tensor as one dense buffer from its data pointer, so a
        # strided view handed to it would be read wrong; every path that wraps one makes it
        # contiguous first, and this keeps it that way.
        raise ValueError("only a contiguous tensor can be handed to .NET")
    on_host = value.device.type == "cpu"
    return (
        KIND_TENSOR,
        _CODES[value.dtype],
        list(value.shape),
        on_host,
        value.data_ptr(),
        value.numel() * value.element_size(),
        -1 if on_host else (value.device.index if value.device.index is not None else torch.cuda.current_device()),
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


def _read_on_run_device(value, run_device):
    """An argument as the run computes on it. A tensor is fed on the run's device already; a
    sequence is fed in host memory, where a run reads it, and its elements are read onto the run's
    device as the run reads them."""
    if isinstance(value, list):
        return [item.to(run_device) if isinstance(item, torch.Tensor) else item for item in value]
    return value


def _export(value, device, taken, ids):
    """An output made safe to hand over: contiguous, on `device` -- the run's device for a tensor,
    the host for a sequence's elements, which is where the .NET side leaves each -- and in memory of
    its own -- never an input's, a constant's or another output's, since the caller owns what it is
    handed and may write to it."""
    if value is None:
        return None
    if isinstance(value, list):
        return [_export(item, torch.device("cpu"), taken, ids) for item in value]
    if is_strings(value):
        if id(value) in ids:
            value = value.copy()
        ids.add(id(value))
        return value
    value = value.detach()
    if value.device != device:
        value = value.to(device)
    value = value.contiguous()
    storage = _storage_of(value)
    if storage is not None and storage in taken:
        value = value.clone()
        storage = _storage_of(value)
    if storage is not None:
        taken.add(storage)
    return value


def run(main, args, wanted, run_device, constant_storages, constant_ids,
        stop_address=0, severity=None, aliases=(), limit_bytes=-1, shrink=False):
    """Runs a translated model and exports the outputs at indices `wanted`, each tensor on the run's
    device and each sequence in host memory -- where the .NET side reads every input and leaves every
    output. Returns (value, description, aliased) per output.

    `args` are where the run reads them already: a tensor on the run's device, a string tensor or a
    sequence in host memory. `stop_address` is the address of a 32-bit flag the .NET side sets to
    stop the run, or 0 for a run nothing can stop; `severity` the least ONNX log severity a warning
    the run raises is shown at; `aliases` one (output index, input index) per slot of the
    translation's plan, in the numbering its `_alias_write` calls use -- an index -1 where the run may
    not write that slot into a consumed input -- see _Aliasing; `limit_bytes` what the run may
    allocate on a CUDA device beyond what its allocator holds there already, or -1; `shrink` whether
    to hand the device's unused cached blocks back once the run is over."""
    run_device = torch.device(run_device)
    outputs = moved = None
    try:
        tokens = [_device.set(run_device), _warning_severity.set(severity)]
        capped = None
        try:
            if stop_address:
                tokens.append(_stop_flag.set(ctypes.c_int32.from_address(stop_address)))
            moved = [_read_on_run_device(arg, run_device) for arg in args]
            aliasing = _Aliasing(args, moved, aliases, run_device, constant_storages)
            tokens.append(_aliasing.set(aliasing))
            capped = _cap(run_device, limit_bytes)
            with torch.no_grad():
                outputs = main(*moved)
        finally:
            for token in reversed(tokens):
                token.var.reset(token)
            _uncap(capped)
        return _export_all(outputs, args, moved, wanted, run_device, aliasing, constant_storages, constant_ids)
    finally:
        # After the outputs are exported and every intermediate is dropped, so that what the run
        # allocated and no longer holds is handed back too, and on a failed run as well.
        outputs = moved = None
        if shrink and run_device.type == "cuda":
            torch.cuda.empty_cache()


def _export_all(outputs, args, moved, wanted, run_device, aliasing, constant_storages, constant_ids):
    """(value, description, aliased) per wanted output: see `run`."""
    taken = set(constant_storages) | storages(args) | storages(moved)
    ids = set(constant_ids) | {id(arg) for arg in args}
    results = [None] * len(wanted)
    into = {}
    for position, index in enumerate(wanted):
        slot = aliasing.slot_of(index, outputs[index])
        if slot is not None and slot not in into:
            into[slot] = position
            continue
        value = _export(outputs[index], run_device, taken, ids)
        results[position] = (value, describe(value), False)
    # Last, so that every other output has been read out of the consumed memory before one is
    # handed over in it: an output that is a view of a consumed input was cloned from it above.
    for slot, position in into.items():
        index = wanted[position]
        value = aliasing.export(slot, outputs[index])
        if value is None:
            value = _export(outputs[index], run_device, taken, ids)
            results[position] = (value, describe(value), False)
            continue
        storage = _storage_of(value)
        if storage is not None:
            taken.add(storage)
        results[position] = (value, describe(value), True)
    return results


# ---- models ----------------------------------------------------------------------------------

_compiled = collections.OrderedDict()
_COMPILED_KEPT = 64


def load_model(source, filename, constants):
    """The `main` function of a translated model, with `constants` bound. The compiled code is
    cached by `filename`, which names the hash of `source`; the constants are the session's own.

    The cache keeps the models loaded last, and the source lines tracebacks show for them: a session
    holds its own code, so one whose code has left the cache runs on, its tracebacks without lines."""
    code = _compiled.get(filename)
    if code is None:
        code = compile(source, filename, "exec")
        _compiled[filename] = code
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        while len(_compiled) > _COMPILED_KEPT:
            evicted, _ = _compiled.popitem(last=False)
            linecache.cache.pop(evicted, None)
    else:
        _compiled.move_to_end(filename)
    namespace = {"__name__": "shorokoo_model", "_C": constants, "_stop": stop_point, "_alias_write": alias_write}
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


def torch_cuda_version():
    """The CUDA version torch was built for, or "" for a build without CUDA."""
    return torch.version.cuda or ""


# ---- stopping a run --------------------------------------------------------------------------

class RunStopped(Exception):
    """A run was stopped between two nodes, because the .NET side asked it to stop."""


_stop_flag = contextvars.ContextVar("shorokoo_stop_flag", default=None)


def stop_point():
    """Stops the run in progress here if it has been asked to stop. A translated model calls this
    before every node; a run nothing can stop has no flag, and pays one lookup."""
    flag = _stop_flag.get()
    if flag is not None and flag.value:
        raise RunStopped("the run was stopped before it finished")


# ---- warnings --------------------------------------------------------------------------------

WARNING = 2
_warning_severity = contextvars.ContextVar("shorokoo_warning_severity", default=None)
_show_warning = warnings.showwarning


def _show_warning_at_severity(message, category, filename, lineno, file=None, line=None):
    """Shows a warning unless the run that raised it asked only for errors: the session's log
    severity, read per run, since the interpreter's warning filters are the whole process's."""
    severity = _warning_severity.get()
    if severity is not None and severity > WARNING:
        return
    _show_warning(message, category, filename, lineno, file, line)


if getattr(warnings.showwarning, "__name__", "") != "_show_warning_at_severity":
    warnings.showwarning = _show_warning_at_severity


# ---- writing an output into a consumed input -------------------------------------------------

_WRITABLE = (torch.float32, torch.float64, torch.float16, torch.bfloat16)
_WRITERS = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}
_aliasing = contextvars.ContextVar("shorokoo_aliasing", default=None)


class _Aliasing:
    """The consumed inputs a run may write outputs into, per output slot, and what it wrote.

    A slot's input is a target only where it is memory of its own: a torch tensor fed at that one
    position on the run's device, in no other argument's or constant's storage. The node that
    produces the output writes it there (`alias_write`), and the output is handed over in it."""

    def __init__(self, args, moved, aliases, run_device, constant_storages):
        count = len(aliases)
        self.output_index = [index for index, _ in aliases]
        self.in_graph = [None] * count
        self.written = [None] * count
        if not count:
            return
        held = {}
        for value in list(args) + list(moved):
            if isinstance(value, torch.Tensor):
                storage = _storage_of(value)
                if storage is not None:
                    held[storage] = held.get(storage, 0) + 1
        constants = set(constant_storages)
        for slot, (_, index) in enumerate(aliases):
            if index < 0 or index >= len(args) or not isinstance(args[index], torch.Tensor):
                continue
            arg, fed = args[index], moved[index]
            storage = _storage_of(arg)
            if storage is None or storage in constants or fed is not arg or held[storage] != 2:
                continue
            if arg.device == run_device:
                self.in_graph[slot] = arg

    def slot_of(self, index, value):
        """The slot output `value`, at output index `index`, is to be handed over in: one whose
        target the graph wrote it into. Else None."""
        for slot, output in enumerate(self.output_index):
            if output != index:
                continue
            if self.written[slot] is not None and value is self.written[slot]:
                return slot
        return None

    def export(self, slot, value):
        """The output of `slot` as handed over, in its target's memory; None where it cannot be."""
        if self.written[slot] is not None:
            return value.detach()
        return None


def _same_layout(a, b):
    return a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride()


def _shares(value, storage):
    if isinstance(value, list):
        return any(_shares(item, storage) for item in value)
    return isinstance(value, torch.Tensor) and _storage_of(value) == storage


def alias_write(slot, op, a, b, live):
    """Output slot `slot`, `op(a, b)`, written into the consumed input the run may write it into,
    and returned; or None where that cannot be done, for the caller to compute it as usual.

    It is done only where the result is exactly what the ordinary node computes, in the memory of an
    input nothing reads any more: the target a floating-point tensor on the operands' device, of
    their one dtype, and the shape they broadcast to; neither operand in its memory -- except the
    first as the target itself, element for element, which is how the operator writes in place --
    and none of `live`, the values the rest of the run still reads that could be views of it."""
    state = _aliasing.get()
    if state is None or slot >= len(state.in_graph):
        return None
    target = state.in_graph[slot]
    if target is None:
        return None
    state.in_graph[slot] = None
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return None
    dtype = target.dtype
    if dtype not in _WRITABLE or a.dtype != dtype or b.dtype != dtype:
        return None
    if a.device != target.device or b.device != target.device:
        return None
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return None
    if tuple(torch.broadcast_shapes(a.shape, b.shape)) != tuple(target.shape):
        return None
    storage = _storage_of(target)
    if storage is None:
        return None
    if _storage_of(a) == storage and not _same_layout(a, target):
        return None
    if _storage_of(b) == storage or any(_shares(value, storage) for value in live):
        return None
    _WRITERS[op](a, b, out=target)
    state.written[slot] = target
    return target


# ---- device memory ---------------------------------------------------------------------------

def _cap(run_device, limit_bytes):
    """Caps what torch's CUDA caching allocator may hold on the run's device at what it holds there
    now plus `limit_bytes`, for the length of the run, and returns what to restore; None where there
    is nothing to cap. The allocator is the whole process's, so the cap is too.

    What it holds, not what it has handed out: the fraction caps the memory it reserves from the
    device, and a segment only partly handed out is reserved all the same."""
    if limit_bytes is None or limit_bytes < 0 or run_device.type != "cuda":
        return None
    index = run_device.index if run_device.index is not None else torch.cuda.current_device()
    # The fraction caps what the allocator reserves from the device, and a block it already holds
    # cached serves an allocation past the cap: handing those back first is what makes the cap the
    # run's. A budgeted run hands them back as it ends too, so there is seldom anything to hand back.
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(index).total_memory
    allowed = torch.cuda.memory_reserved(index) + limit_bytes
    previous = _memory_fraction(index)
    torch.cuda.set_per_process_memory_fraction(min(1.0, allowed / total), index)
    return (index, previous)


def _uncap(capped):
    if capped is not None:
        index, previous = capped
        torch.cuda.set_per_process_memory_fraction(previous, index)


def _memory_fraction(index):
    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    try:
        return float(getter(index)) if getter is not None else 1.0
    except Exception:
        return 1.0


def arena_statistics(device_name):
    """The CUDA caching allocator's figures for a device, in the order ArenaStatistics takes them
    after the limit: in use, largest single allocation (which torch does not record: 0), peak in
    use, allocations, segments held, segments released, reserves (none: 0), reserved in all, and
    the bytes callers requested of what is in use, without the allocator's rounding."""
    device = torch.device(device_name)
    if device.type != "cuda":
        return None
    stats = torch.cuda.memory_stats(device)
    get = lambda key: int(stats.get(key, 0))
    return (
        get("allocated_bytes.all.current"),
        0,
        get("allocated_bytes.all.peak"),
        get("allocation.all.allocated"),
        get("segment.all.current"),
        get("segment.all.freed"),
        0,
        get("reserved_bytes.all.current"),
        get("requested_bytes.all.current") or get("allocated_bytes.all.current"),
    )
