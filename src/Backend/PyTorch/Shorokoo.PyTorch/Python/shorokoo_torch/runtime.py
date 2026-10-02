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
        stop_address=0, severity=None, aliases=(), limit_bytes=-1, shrink=False, tensor_float32=False,
        placements=(), writable=()):
    """Runs a translated model and exports the outputs at indices `wanted`, each tensor on the run's
    device and each sequence in host memory -- where the .NET side reads every input and leaves every
    output. Returns (value, description, written) per output: `written` is True for an output
    written into a consumed input by an alias pair, a one-element tuple (slot,) for one written into
    its placement's range, and False otherwise.

    `args` are where the run reads them already: a tensor on the run's device, a string tensor or a
    sequence in host memory. `stop_address` is the address of a 32-bit flag the .NET side sets to
    stop the run, or 0 for a run nothing can stop; `severity` the least ONNX log severity a warning
    the run raises is shown at; `aliases` one (output index, input index) per slot of the
    translation's plan, in the numbering its `_alias_write` calls use -- an index -1 where the run may
    not write that slot into a consumed input -- see _Aliasing; `limit_bytes` what the run may
    allocate on a CUDA device beyond what its allocator holds there already, or -1; `shrink` whether
    to hand the device's unused cached blocks back once the run is over; `tensor_float32` whether a
    run on a CUDA device may compute float32 products, convolutions and recurrent layers in
    TensorFloat-32 -- see float32_precision; `placements` the ranges of consumed inputs the
    translation's `_into` calls write values into, one per slot -- see _Placing; `writable` the
    indices of the inputs the run consumed, which its `_over` calls may write over -- see
    _Writing."""
    run_device = torch.device(run_device)
    if run_device.type == "cuda":
        float32_precision(tensor_float32)
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
            placing = _Placing(args, moved, placements, run_device, constant_storages)
            tokens.append(_placing.set(placing))
            tokens.append(_writing.set(_Writing(args, moved, writable, run_device, constant_storages)))
            capped = _cap(run_device, limit_bytes)
            with torch.no_grad():
                outputs = main(*moved)
        finally:
            for token in reversed(tokens):
                token.var.reset(token)
            _uncap(capped)
        return _export_all(outputs, args, moved, wanted, run_device, aliasing, placing, constant_storages, constant_ids)
    finally:
        # After the outputs are exported and every intermediate is dropped, so that what the run
        # allocated and no longer holds is handed back too, and on a failed run as well.
        outputs = moved = None
        if shrink and run_device.type == "cuda":
            torch.cuda.empty_cache()


def float32_precision(tensor_float32):
    """Sets whether cuBLAS products and cuDNN convolutions and recurrent layers of float32 operands may
    be computed in TensorFloat-32, which rounds each operand's significand to 11 bits; off, they are
    computed in full float32 precision. torch's switches for it are the whole process's and are read
    as each kernel is launched, so a run on a card sets them from its session as it starts, and the
    .NET side keeps runs that set them differently apart. They are set through allow_tf32, which also
    sets torch's fp32_precision to match; setting fp32_precision alone would leave the two
    disagreeing, and torch then refuses to read allow_tf32."""
    torch.backends.cuda.matmul.allow_tf32 = bool(tensor_float32)
    torch.backends.cudnn.allow_tf32 = bool(tensor_float32)


def _export_all(outputs, args, moved, wanted, run_device, aliasing, placing, constant_storages, constant_ids):
    """(value, description, written) per wanted output: see `run`."""
    taken = set(constant_storages) | storages(args) | storages(moved)
    ids = set(constant_ids) | {id(arg) for arg in args}
    results = [None] * len(wanted)
    into = {}
    for position, index in enumerate(wanted):
        slot = aliasing.slot_of(index, outputs[index])
        if slot is not None and slot not in into:
            into[slot] = position
            continue
        placed = placing.slot_of(index, outputs[index])
        if placed is not None and placed not in placing.handed:
            # Handed over where the run wrote it, in the consumed input's memory: the .NET side
            # makes it a value standing on that input's block.
            placing.handed.add(placed)
            value = outputs[index].detach()
            results[position] = (value, describe(value), (placed,))
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
    namespace = {"__name__": "shorokoo_model", "_C": constants, "_stop": stop_point, "_alias_write": alias_write,
                 "_into": place_into, "_over": write_over}
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


# ---- writing a run's values into the memory it consumed ---------------------------------------

_FLOATING = (torch.float32, torch.float64, torch.float16, torch.bfloat16)
_placing = contextvars.ContextVar("shorokoo_placing", default=None)
_fast = None


class _Placing:
    """The ranges of consumed inputs a run writes values into, one per slot of the translation's
    placements, and what it wrote.

    A slot is (input index, byte offset, byte count, element type code, shape, output index): the
    value goes `byte count` bytes from byte `offset` of the memory of `main`'s input `input index`,
    and is graph output `output index` (-1 for none). Which ranges are safe to write, and when, the
    .NET side has proved over the graph; what is checked here is that each input is a tensor of
    memory of its own: fed at that one position on the run's device, contiguous, its bytes no other
    argument's and no constant's. A slot whose input is not is not written, and its value computed
    as the plain translation computes it."""

    def __init__(self, args, moved, placements, run_device, constant_storages):
        self.slots = list(placements)
        self.targets = [None] * len(self.slots)
        self.written = [None] * len(self.slots)
        self.handed = set()
        self.blocks = {}
        if not self.slots:
            return
        extents = []
        for index, value in enumerate(list(args) + list(moved)):
            for extent in _extents(value):
                extents.append((index % len(args), extent))
        constants = set(constant_storages)
        for index in {slot[0] for slot in self.slots}:
            if index < 0 or index >= len(args):
                continue
            arg = args[index]
            if not isinstance(arg, torch.Tensor) or moved[index] is not arg or arg.device != run_device:
                continue
            if not arg.is_contiguous() or arg.numel() == 0 or _storage_of(arg) in constants:
                continue
            start = arg.data_ptr()
            end = start + arg.numel() * arg.element_size()
            if any(other != index and lo < end and start < hi for other, (lo, hi) in extents):
                continue
            try:
                self.blocks[index] = (arg.detach().reshape(-1).view(torch.uint8), end - start)
            except RuntimeError:
                continue

    def target(self, slot):
        """The tensor over slot `slot`'s range, or None where the slot is not written."""
        if slot >= len(self.slots):
            return None
        target = self.targets[slot]
        if target is None:
            index, offset, count, code, shape, _ = self.slots[slot]
            block = self.blocks.get(index)
            if block is None or offset < 0 or offset + count > block[1]:
                return None
            target = block[0][offset:offset + count].view(_DTYPES[code]).view(shape)
            self.targets[slot] = target
        return target

    def slot_of(self, index, value):
        """The slot output `value`, at output index `index`, was written into; else None."""
        for slot, placement in enumerate(self.slots):
            if placement[5] == index and self.written[slot] is not None and value is self.written[slot]:
                return slot
        return None


def _extents(value):
    """The address ranges of memory `value` reads: a contiguous tensor's own bytes, all of a strided
    view's storage, and each element's of a sequence."""
    if isinstance(value, list):
        return [extent for item in value for extent in _extents(item)]
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return []
    if value.is_contiguous():
        start = value.data_ptr()
        return [(start, start + value.numel() * value.element_size())]
    storage = value.untyped_storage()
    return [(storage.data_ptr(), storage.data_ptr() + storage.nbytes())]


def _same_memory(a, b):
    return (a.data_ptr() == b.data_ptr() and a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and a.stride() == b.stride())


def _overlap(a, b):
    return any(lo < end and start < hi for lo, hi in _extents(a) for start, end in _extents(b))


def writes_into(out, like):
    """Whether a support function handed `out` (its `_out`) for a result laid out as `like` -- the
    result's type, shape and device -- writes the result there: `out` is a tensor of exactly that
    layout. A function that does not computes its result as it would without one."""
    return (isinstance(out, torch.Tensor) and isinstance(like, torch.Tensor) and out.dtype == like.dtype
            and tuple(out.shape) == tuple(like.shape) and out.device == like.device)


def _fast_writers():
    """Per support function, how it writes its result into a tensor it is handed allocating nothing,
    exactly as it computes it: ("out", f) for torch's f taking out=, over floating-point operands;
    ("relu", None) for the activation torch has only an in-place form of; ("matmul", f) for a
    product of two matrices or stacks of them, over floating-point operands; ("fill", None) for
    constant_of_shape; ("cat", None) for concat, part by part; ("own", None) for a function that
    takes the tensor itself as `_out` and writes its result there, step by step, allocating nothing
    a plain call does not allocate too (`writes_into`)."""
    global _fast
    if _fast is None:
        from . import ops_conv_pool as cp, ops_elementwise as e, ops_linalg as la, ops_logic as lo, ops_norm as n, ops_shape as s
        table = {function: ("out", op) for function, op in [
            (e.neg, torch.neg), (e.abs_, torch.abs), (e.sigmoid, torch.sigmoid), (e.exp, torch.exp),
            (e.log, torch.log), (e.sqrt, torch.sqrt), (e.tanh, torch.tanh), (e.sin, torch.sin),
            (e.cos, torch.cos), (e.tan, torch.tan), (e.asin, torch.asin), (e.acos, torch.acos),
            (e.atan, torch.atan), (e.sinh, torch.sinh), (e.cosh, torch.cosh), (e.asinh, torch.asinh),
            (e.acosh, torch.acosh), (e.atanh, torch.atanh), (e.reciprocal, torch.reciprocal),
            (e.floor, torch.floor), (e.ceil, torch.ceil), (e.round_, torch.round), (e.erf, torch.erf),
            (e.add, torch.add), (e.sub, torch.sub), (e.mul, torch.mul), (e.div, torch.div),
            (e.pow_, torch.pow), (e.max_, torch.maximum), (e.min_, torch.minimum), (e.sum_, torch.add),
        ]}
        table[e.relu] = ("relu", None)
        for function in (e.clip, e.softmax, e.log_softmax, e.gelu, n.layer_normalization, n.batch_normalization,
                         cp.conv, la.gemm, lo.where):
            table[function] = ("own", None)
        table[la.matmul] = ("matmul", torch.matmul)
        table[s.constant_of_shape] = ("fill", None)
        table[s.concat] = ("cat", None)
        _fast = table
    return _fast


def _write_fast(fast, target, args, kwargs):
    """Writes the call into `target` allocating nothing, where `fast` can; whether it did."""
    kind, op = fast
    dtype = target.dtype
    if kind == "fill":
        value = kwargs.get("value")
        if len(args) != 1 or set(kwargs) - {"value"} or (value.dtype if value is not None else torch.float32) != dtype:
            return False
        if [int(d) for d in args[0].reshape(-1).tolist()] != list(target.shape):
            return False
        target.fill_(value.reshape(-1)[0].item() if value is not None else 0)
        return True
    if kind == "cat":
        if set(kwargs) != {"axis"} or not args or target.dim() == 0:
            return False
        rank = target.dim()
        axis = kwargs["axis"] % rank
        if not all(isinstance(part, torch.Tensor) and part.dtype == dtype and part.device == target.device
                   and part.dim() == rank for part in args):
            return False
        if sum(part.shape[axis] for part in args) != target.shape[axis]:
            return False
        if any(part.shape[d] != target.shape[d] for part in args for d in range(rank) if d != axis):
            return False
        start = 0
        for part in args:
            length = part.shape[axis]
            destination = target.narrow(axis, start, length)
            if length and not _same_memory(part, destination):
                destination.copy_(part)
            start += length
        return True
    if kind == "matmul":
        if kwargs or len(args) != 2 or dtype not in _FLOATING:
            return False
        a, b = args
        if not all(isinstance(t, torch.Tensor) and t.dtype == dtype and t.device == target.device and t.dim() >= 2 for t in args):
            return False
        if a.shape[-1] != b.shape[-2]:
            return False
        expected = tuple(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])) + (a.shape[-2], b.shape[-1])
        if expected != tuple(target.shape):
            return False
        op(a, b, out=target)
        return True
    if dtype not in _FLOATING or kwargs or not args:
        return False
    device, shape, whole = target.device, target.shape, True
    for a in args:
        if not isinstance(a, torch.Tensor) or a.dtype != dtype or (a.device != device and a.dim()):
            return False
        whole = whole and a.shape == shape
    if not whole and torch.broadcast_shapes(*(a.shape for a in args)) != shape:
        return False
    if kind == "relu":
        if not _same_memory(args[0], target):
            target.copy_(args[0])
        torch.relu_(target)
        return True
    op(*args, out=target)
    return True


def _write_copy(target, value):
    """Copies `value`, a result already computed, into `target`; whether it is there now."""
    if not isinstance(value, torch.Tensor) or value.dtype != target.dtype or tuple(value.shape) != tuple(target.shape):
        return False
    if value.device != target.device:
        return False
    if _same_memory(value, target):
        return True
    if _overlap(value, target):
        return False
    target.copy_(value)
    return True


_writing = contextvars.ContextVar("shorokoo_writing", default=None)


class _Writing:
    """Which of a run's inputs its `_over` calls may write over: those it consumed (`writable`,
    by index), fed at that one position on the run's device, in no other argument's or constant's
    storage. Every other input is the caller's, or another's too, and is only read."""

    def __init__(self, args, moved, writable, run_device, constant_storages):
        self.inputs = {id(value) for value in moved if isinstance(value, torch.Tensor)}
        self.own = set()
        held = {}
        for value in list(args) + list(moved):
            if isinstance(value, torch.Tensor) and (storage := _storage_of(value)) is not None:
                held[storage] = held.get(storage, 0) + 1
        constants = set(constant_storages)
        for index in writable:
            if index < 0 or index >= len(args) or not isinstance(args[index], torch.Tensor):
                continue
            arg = args[index]
            storage = _storage_of(arg)
            if (moved[index] is arg and arg.device == run_device and storage is not None
                    and storage not in constants and held[storage] == 2):
                self.own.add(id(arg))

    def may_write(self, target):
        return id(target) not in self.inputs or id(target) in self.own


def write_over(target, function, *args, **kwargs):
    """`function(*args, **kwargs)` -- a node of the translated graph that computes the same over its
    operand -- written over `target`, an operand of it that nothing reads after it, and returned
    there, where torch writes the result into it (_fast_writers) and the run computes no gradient;
    elsewhere, or where the result is not of the operand's type and shape, the result as computed.
    An input of the run is written over only where the run consumed it (_Writing)."""
    writing = _writing.get()
    if (isinstance(target, torch.Tensor) and writing is not None and writing.may_write(target)
            and not torch.is_grad_enabled() and not target.requires_grad
            and target.is_contiguous() and target.numel() > 0):
        fast = _fast_writers().get(function)
        if fast is not None and fast[0] != "own" and _write_fast(fast, target, args, kwargs):
            return target
        if fast is not None and fast[0] == "own":
            return function(*args, _out=target, **kwargs)
    return function(*args, **kwargs)


def place_into(slot, own, function, *args, **kwargs):
    """`function(*args, **kwargs)` -- a node of the translated graph -- written into the range of
    placement slot `slot` and returned there, where the run hands one over: by torch writing the
    result into it where `function` has a form that does (_fast_writers), else by copying the result
    in. Where the run hands over none, or the result cannot go there, the result as computed --
    copied into memory of its own where `own` says the placement was proved with the value out of
    its operands' memory and it is still in it. A function returning a tuple -- a node whose first
    output alone is used -- has that first element placed, and the tuple returned."""
    state = _placing.get()
    target = state.target(slot) if state is not None else None
    fast = _fast_writers().get(function) if target is not None else None
    if fast is not None and fast[0] != "own" and _write_fast(fast, target, args, kwargs):
        state.written[slot] = target
        return target
    value = function(*args, _out=target, **kwargs) if fast is not None else function(*args, **kwargs)
    rest = None
    if isinstance(value, tuple):
        value, rest = value[0], value[1:]
    written = False
    if target is not None:
        if isinstance(value, torch.Tensor) and _same_memory(value, target):
            written = True
        elif _write_copy(target, value):
            written, value = True, target
        if written:
            state.written[slot] = value
    if not written:
        if own and isinstance(value, torch.Tensor):
            storage = _storage_of(value)
            if storage is not None and storage in storages([a for a in args if isinstance(a, (torch.Tensor, list))]):
                value = value.clone(memory_format=torch.contiguous_format)
    return value if rest is None else (value,) + rest


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
