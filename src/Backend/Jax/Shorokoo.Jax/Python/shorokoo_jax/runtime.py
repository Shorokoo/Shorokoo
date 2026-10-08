"""The runtime half of Shorokoo's JAX backend: values in and out of the .NET side, compiling a
translated model with XLA, running it, and what every operator family shares.

Values are numpy arrays in host memory and jax arrays in a device's memory. The .NET side reads a
host value through its buffer, so a host value handed to it is always a numpy array of its own --
C-contiguous, writable, and no other value's memory. Element types travel as ONNX
TensorProto.DataType codes. JAX has no string tensors and no sequences, and the translation refuses
a model that uses either before anything here runs.

JAX is started with 64-bit types enabled: ONNX's int64 and float64 are 64 bits, and JAX's default
would compute them in 32. That is a setting of the whole process's JAX.
"""

import collections
import contextlib
import contextvars
import ctypes
import linecache
import os
import threading
import warnings

# Before JAX is imported: a CUDA device's memory is taken as it is needed rather than 75 % of the
# card up front, since the card is shared with ONNX Runtime and PyTorch in the same process. A value
# the program's environment already sets is kept.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

jax.config.update("jax_enable_x64", True)

_DTYPES = {
    1: np.dtype(np.float32),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int8),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int16),
    6: np.dtype(np.int32),
    7: np.dtype(np.int64),
    9: np.dtype(np.bool_),
    10: np.dtype(np.float16),
    11: np.dtype(np.float64),
    12: np.dtype(np.uint32),
    13: np.dtype(np.uint64),
    14: np.dtype(np.complex64),
    15: np.dtype(np.complex128),
    16: np.dtype(jnp.bfloat16),
    17: np.dtype(jnp.float8_e4m3fn),
    18: np.dtype(jnp.float8_e4m3fnuz),
    19: np.dtype(jnp.float8_e5m2),
    20: np.dtype(jnp.float8_e5m2fnuz),
}
_CODES = {dtype: code for code, dtype in _DTYPES.items()}

FLOAT8 = (_DTYPES[17], _DTYPES[18], _DTYPES[19], _DTYPES[20])

KIND_TENSOR = 0

# The precision of the products and convolutions of the program being traced: the model's own,
# which the .NET side settles from the session's PrecisionSettings -- HIGHEST, full float32 precision,
# unless a session on a card allows TensorFloat-32, which XLA computes at HIGH. XLA's default on a
# card rounds the operands of a float32 product to TensorFloat-32, so every product and convolution
# names its precision rather than leaving it to that default.
_precision = contextvars.ContextVar("shorokoo_jax_precision", default=jax.lax.Precision.HIGHEST)


def precision():
    """The precision a product or convolution of the program being traced is computed in."""
    return _precision.get()


def jax_dtype(code):
    """The dtype of an ONNX element type code; string and 4-bit types have none."""
    try:
        return _DTYPES[int(code)]
    except KeyError:
        raise NotImplementedError(f"ONNX element type {code} has no JAX dtype") from None


def dtype_code(value):
    """The ONNX element type code of a value or a dtype."""
    dtype = value if isinstance(value, np.dtype) else np.dtype(value.dtype)
    return _CODES[dtype]


def is_floating(value):
    """Whether a value (or a dtype) is of a floating-point type, the 8- and 16-bit ones included."""
    dtype = value if isinstance(value, np.dtype) else np.dtype(value.dtype)
    return jnp.issubdtype(dtype, jnp.floating)


def is_integral(value):
    dtype = value if isinstance(value, np.dtype) else np.dtype(value.dtype)
    return jnp.issubdtype(dtype, jnp.integer) or dtype == np.bool_


# ---- concrete and traced values ---------------------------------------------------------------

class DataDependentShape(NotImplementedError):
    """An operator needs a number -- a shape, a count, an axis -- that the graph computes from the
    values of an input, which a program compiled once for every value of its inputs cannot have; or
    otherwise makes a shape its inputs' values decide. `operator` names it; the .NET side refuses
    the model naming it."""

    def __init__(self, operator, what, message=None):
        super().__init__(message or (
            f"The {operator} operator needs {what} as a number when the model is compiled, and the graph "
            f"computes it from the values of an input; the JAX backend compiles a model once per input "
            f"shape, so it cannot run it"))
        self.operator = operator


def concrete(*values):
    """Whether every value is known while the model is traced: a numpy array or scalar, a Python
    number, or None -- anything but a jax array or a tracer."""
    return not any(isinstance(v, jax.Array) for v in values)


# Types numpy computes in without widening -- summing, multiplying and rounding in the narrow type
# itself -- where XLA and torch accumulate wider. A value of one is never a shape, so it is left to
# jax.numpy even where it is known, and the concrete path cannot give another answer than the traced.
_NARROW_FLOATS = {np.dtype(np.float16), np.dtype(jnp.bfloat16), *(np.dtype(t) for t in
                  (jnp.float8_e4m3fn, jnp.float8_e4m3fnuz, jnp.float8_e5m2, jnp.float8_e5m2fnuz))}


def xp(*values):
    """numpy where every value is concrete, so the result is too; jax.numpy otherwise, and for any
    value of a floating-point type narrower than 32 bits."""
    if not concrete(*values):
        return jnp
    narrow = any(getattr(v, "dtype", None) in _NARROW_FLOATS for v in values)
    return jnp if narrow else np


def divide(a, b):
    """a / b, rounded as a division is. XLA rewrites a division by a value it knows when it compiles
    -- a constant, or anything it computes from constants alone, however it got there -- into a
    multiplication by its reciprocal, which can land one ulp off the quotient, and a rounding after it
    (QuantizeLinear, a pooling bin's edge) on another integer; so the divisor is hidden from that
    rewrite."""
    if concrete(a, b):
        return xp(a, b).divide(a, b)
    # Hidden at the shape it divides at: a scalar hidden and then broadcast is rewritten all the same.
    b = jnp.asarray(b, dtype=getattr(b, "dtype", None))
    b = jnp.broadcast_to(b, jnp.broadcast_shapes(jnp.shape(a), b.shape))
    return jnp.divide(a, jax.lax.optimization_barrier(b))


def ints(value, operator, what):
    """A concrete integer tensor's elements as Python ints, for an operator that needs them as
    numbers; DataDependentShape where the value is traced."""
    if not concrete(value):
        raise DataDependentShape(operator, what)
    return [int(v) for v in np.asarray(value).reshape(-1).tolist()]


def number(value, operator, what):
    """A concrete scalar's value as a Python number; DataDependentShape where it is traced."""
    if not concrete(value):
        raise DataDependentShape(operator, what)
    return np.asarray(value).reshape(-1)[0].item()


def detach(value):
    """`value` with no gradient flowing back through it."""
    return value if concrete(value) else jax.lax.stop_gradient(value)


def asarray(value, dtype=None):
    """A value as an array of the module `xp` picks for it."""
    return xp(value).asarray(value, dtype=dtype)


# ---- values in and out ------------------------------------------------------------------------

def device_of(name):
    """The jax device a .NET device name (cpu, cuda:N) names."""
    if name == "cpu":
        return jax.devices("cpu")[0]
    platform, _, index = name.partition(":")
    return jax.devices(platform)[int(index or 0)]


def from_host(address, nbytes, code, shape, device_name):
    """A value holding a copy of `nbytes` bytes at host `address`: a numpy array of its own on the
    host, or a jax array on a device."""
    array = np.empty(tuple(shape), dtype=jax_dtype(code))
    if nbytes:
        ctypes.memmove(array.ctypes.data, address, nbytes)
    if device_name != "cpu":
        return jax.device_put(array, device_of(device_name))
    return array


# The bytes of the one host buffer the .NET side streams a save or a load through
# (StagedReadBack.StagingBytes): an array no larger is moved whole, through host memory that size,
# compiling nothing; only a larger one is moved by the piece, through the programs below.
_STAGING_BYTES = 8 << 20


def empty(code, shape, device_name):
    """A tensor whose contents are unspecified: zeros, since a jax array is never uninitialized. An
    array on a device no larger than the staging buffer is sent there from host zeros, compiling
    nothing; a larger one is made on the device, never through host memory, at the cost of one
    program per shape and type."""
    dtype = jax_dtype(code)
    if device_name == "cpu":
        return np.zeros(tuple(shape), dtype=dtype)
    if int(np.prod(shape, dtype=np.int64)) * dtype.itemsize <= _STAGING_BYTES:
        return jax.device_put(np.zeros(tuple(shape), dtype=dtype), device_of(device_name))
    return jnp.zeros(tuple(shape), dtype=dtype, device=device_of(device_name))


def host_copy(value):
    """A host value of its own with `value`'s contents: C-contiguous and writable."""
    array = np.array(value, copy=True, order="C")
    return array if array.flags.writeable else array.copy()


# A device array's elements `first` to `first + size` as an array of their own, and the array with
# those elements replaced, written over the array handed in (donated): XLA reshapes an array to a
# flat one in place, so neither copies the rest of it. Each compiles a program per array shape and
# type and per size: _piece is only asked for a power of two of elements (_fetch), so it keeps a
# handful per array; _with_piece is asked for a load's pieces, which are of one size and the last.
_piece = jax.jit(lambda array, first, size: jax.lax.dynamic_slice(array.reshape(-1), (first,), (size,)),
                 static_argnums=2)
_with_piece = jax.jit(
    lambda array, piece, first: jax.lax.dynamic_update_slice(array.reshape(-1), piece, (first,)).reshape(array.shape),
    donate_argnums=0)

# The fewest elements _fetch slices off a device array, so that pieces of every size under it share
# one program.
_LEAST_PIECE = 4096


def _elements_of(array, byte_offset, count):
    """The elements covering `count` bytes `byte_offset` bytes into `array`: the first, how many, and
    where the bytes start in them."""
    itemsize = np.dtype(array.dtype).itemsize
    first = byte_offset // itemsize
    return first, -(-(byte_offset + count) // itemsize) - first, byte_offset - first * itemsize


def _fetch(array, first, size):
    """Elements `first` to `first + size` of a device array, in host memory: the whole array copied
    home where they are most of it -- which holds no copy of it on the device -- and otherwise a
    slice holding them of a power of two of elements, and at least _LEAST_PIECE, so that the pieces
    of a gather's many run lengths share a few programs and each holds at most twice its size on
    the device."""
    length = int(np.prod(array.shape, dtype=np.int64))
    elements = max(1 << max(size - 1, 0).bit_length(), _LEAST_PIECE)
    if elements >= length:
        return np.asarray(array).reshape(-1)[first:first + size]
    start = min(first, length - elements)
    return np.asarray(_piece(array, np.int64(start), elements))[first - start:first - start + size]


def copy_range_to_host(array, byte_offset, address, count):
    """Copies `count` bytes of a device array, `byte_offset` bytes in, to host `address`: the
    elements covering them fetched home (_fetch), without the rest of the array where they are
    only part of it."""
    if count:
        first, size, skip = _elements_of(array, byte_offset, count)
        piece = np.ascontiguousarray(_fetch(array, first, size))
        ctypes.memmove(address, piece.ctypes.data + skip, count)


def copy_host_to_range(array, byte_offset, address, count):
    """A device array holding `array`'s contents with `count` bytes at host `address` written
    `byte_offset` bytes in -- `array` itself, written over in place, which leaves the array handed
    in deleted. Only the elements covering those bytes cross to the device, and those of them the
    bytes cover only in part are fetched first."""
    if not count:
        return array
    first, size, skip = _elements_of(array, byte_offset, count)
    itemsize = np.dtype(array.dtype).itemsize
    if skip or (skip + count) % itemsize:
        piece = np.array(_fetch(array, first, size), copy=True)
    else:
        piece = np.empty(size, dtype=array.dtype)
    ctypes.memmove(piece.ctypes.data + skip, address, count)
    device = next(iter(array.devices()))
    return _with_piece(array, jax.device_put(piece, device), np.int64(first))


def describe(value):
    """(kind, element type code, shape, is host, data address, byte count, device id) of a value;
    the device id is -1 for a value in host memory."""
    if isinstance(value, np.ndarray):
        if not value.flags.c_contiguous or not value.flags.writeable:
            # The .NET side reads and writes a host value as one dense buffer from its address.
            raise ValueError("only a contiguous, writable array can be handed to .NET")
        return (KIND_TENSOR, dtype_code(value), list(value.shape), True, value.ctypes.data, value.nbytes, -1)
    device = next(iter(value.devices()))
    return (KIND_TENSOR, dtype_code(value), list(value.shape), False, 0,
            int(np.prod(value.shape, dtype=np.int64)) * np.dtype(value.dtype).itemsize, device.id)


# ---- models -----------------------------------------------------------------------------------

# A constant larger than this many elements is handed to the compiled program as an argument, on
# the session's device, rather than written into it: XLA copes badly with large literals, and
# nothing computes a shape from a constant this large.
_LARGEST_LITERAL = 4096

_code = collections.OrderedDict()
_CODE_KEPT = 64

# The compiled programs a model keeps, one per signature of input shapes and types, as a training
# rig keeps compiled steps: most models see one or a few.
_PROGRAMS_KEPT = 16


class Model:
    """A translated model loaded for one session: its `main`, its constants, and the XLA programs
    compiled from it so far, one per signature of the shapes and element types of its inputs."""

    def __init__(self, main, constants, device, precision):
        self._main = main
        self._constants = constants
        self.device = device
        self.precision = precision
        self._large = [i for i, c in enumerate(constants) if np.size(c) > _LARGEST_LITERAL]
        self._large_values = [jax.device_put(constants[i], device) for i in self._large]
        self._programs = collections.OrderedDict()
        self._lock = threading.Lock()

    def _entry(self, key, large, *args):
        """`main`, traced with a random key of the run and the large constants as arguments."""
        saved = [self._constants[i] for i in self._large]
        for i, value in zip(self._large, large):
            self._constants[i] = value
        token = _keys.set(_Keys(jax.random.wrap_key_data(key)))
        precision_token = _precision.set(self.precision)
        try:
            with np.errstate(all="ignore"):
                return tuple(self._main(*args))
        finally:
            _precision.reset(precision_token)
            _keys.reset(token)
            for i, value in zip(self._large, saved):
                self._constants[i] = value

    def program(self, signature):
        """The program compiled for `signature` -- one (shape, dtype) per input -- compiling it
        first where it has not been."""
        with self._lock:
            program = self._programs.get(signature)
            if program is not None:
                self._programs.move_to_end(signature)
                return program
            sharding = SingleDeviceSharding(self.device)
            specs = [jax.ShapeDtypeStruct(shape, dtype, sharding=sharding) for shape, dtype in signature]
            key = jax.ShapeDtypeStruct((2,), np.uint32, sharding=sharding)
            large = [jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=sharding) for v in self._large_values]
            program = jax.jit(self._entry).lower(key, large, *specs).compile()
            self._programs[signature] = program
            while len(self._programs) > _PROGRAMS_KEPT:
                self._programs.popitem(last=False)
            return program

    def prepare(self, inputs):
        """Compiles the program for `inputs`, one (element type code, shape) per input, ahead of
        the first run -- where the model's inputs have fixed shapes, so that a model that cannot be
        compiled is refused when its session is made."""
        self.program(tuple((tuple(shape), jax_dtype(code)) for code, shape in inputs))

    def __call__(self, args):
        signature = tuple((tuple(a.shape), np.dtype(a.dtype)) for a in args)
        program = self.program(signature)
        moved = [jax.device_put(a, self.device) for a in args]
        return program(_fresh_key(), self._large_values, *moved)


def load_model(source, filename, constants, device_name, precision="HIGHEST"):
    """The model a translation's source defines, with `constants` bound as `_C`, on `device_name`,
    its products and convolutions compiled at `precision` (a jax.lax.Precision name). The source's
    compiled Python code is cached by `filename`, which names the source's hash."""
    code = _code.get(filename)
    if code is None:
        code = compile(source, filename, "exec")
        _code[filename] = code
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        while len(_code) > _CODE_KEPT:
            evicted, _ = _code.popitem(last=False)
            linecache.cache.pop(evicted, None)
    else:
        _code.move_to_end(filename)
    constants = list(constants)
    namespace = {"__name__": "shorokoo_model", "_C": constants}
    exec(code, namespace)
    return Model(namespace["main"], constants, device_of(device_name), jax.lax.Precision[precision])


def prepare(model, inputs, warned=None):
    token = _warnings_collected.set(warned)
    try:
        model.prepare(inputs)
    finally:
        _warnings_collected.reset(token)


def run(model, args, wanted, warned=None):
    """Runs a model and hands over the outputs at indices `wanted`, each where the .NET side reads
    the model's inputs and leaves its outputs: in the device's memory on a card, and a host value of
    its own on the CPU. Returns (value, description) per output. Every output is ready when this
    returns: nothing the run reads or writes is still in flight."""
    token = _warnings_collected.set(warned)
    try:
        outputs = model(args)
    finally:
        _warnings_collected.reset(token)
    on_device = model.device.platform != "cpu"
    results = []
    for index in wanted:
        value = outputs[index]
        value = jax.block_until_ready(value) if on_device else host_copy(value)
        results.append((value, describe(value)))
    return results


def jax_version():
    return str(jax.__version__)


def device_count(platform):
    """How many devices of `platform` JAX sees; zero where it has no backend for it."""
    try:
        return len(jax.devices(platform))
    except RuntimeError:
        return 0


def arena_statistics(device_name):
    """A device allocator's figures, in the order ArenaStatistics takes them: in use, the most it
    may hold, largest single allocation, peak in use, allocations, extensions (not recorded: 0),
    shrinkages (0), reserves (0), held in all, and in use again as what callers requested, since the
    allocator does not report its rounding. None where the device keeps none."""
    if device_name == "cpu":
        return None
    stats = device_of(device_name).memory_stats() or {}
    get = lambda key: int(stats.get(key, 0))
    return (
        get("bytes_in_use"),
        get("bytes_limit") or -1,
        get("largest_alloc_size"),
        get("peak_bytes_in_use"),
        get("num_allocs"),
        0,
        0,
        0,
        get("pool_bytes") or get("bytes_reserved") or get("bytes_in_use"),
        get("bytes_in_use"),
    )


# ---- random draws -----------------------------------------------------------------------------

class _Keys:
    """The keys a run draws from: the run's key folded with the number of the draw, in the order the
    model's draws are traced, and with the iteration number of every compiled loop the draw is
    inside. The run's key itself never changes: a body XLA compiles once (a scanned or while Loop, an
    If's branch) is traced once, so a key it replaced would be the same every iteration, and a value
    of that body's own trace would leak out of it into the draws after it."""

    def __init__(self, key):
        self._key = key
        self._count = 0
        self._iterations = []

    def next(self):
        key = jax.random.fold_in(self._key, self._count)
        self._count += 1
        for iteration in self._iterations:
            key = jax.random.fold_in(key, iteration)
        return key

    @contextlib.contextmanager
    def iteration(self, number):
        self._iterations.append(number)
        try:
            yield
        finally:
            self._iterations.pop()


_keys = contextvars.ContextVar("shorokoo_jax_keys", default=None)


def next_key(seed=None):
    """The key for one draw: from the node's `seed` where it has one, so that its draws are the same
    every run; else a fresh one of the run's."""
    if seed is not None:
        return jax.random.key(int(np.float32(seed).view(np.uint32)))
    return _keys.get().next()


def loop_iteration(number):
    """Marks the draws traced inside it as those of iteration `number` of a loop compiled once."""
    keys = _keys.get()
    return keys.iteration(number) if keys is not None else contextlib.nullcontext()


def _fresh_key():
    return np.frombuffer(os.urandom(8), dtype=np.uint32).copy()


# ---- warnings ---------------------------------------------------------------------------------

_warnings_collected = contextvars.ContextVar("shorokoo_jax_warnings_collected", default=None)
_show_warning = warnings.showwarning


def _collect_jax_warning(message, category, filename, lineno, file=None, line=None):
    """Appends a warning a call raises to the list the .NET side handed that call, as (category,
    message, location), rather than showing it: the .NET side delivers it to the call's own log
    settings. The interpreter's warning machinery is the whole process's, so the list is read per
    call. A warning raised outside a call goes on to whatever showed warnings when this hook was
    installed, another backend's hook included."""
    collected = _warnings_collected.get()
    if collected is None:
        _show_warning(message, category, filename, lineno, file, line)
        return
    collected.append((getattr(category, "__name__", str(category)), str(message), f"{filename}:{lineno}"))


if getattr(warnings.showwarning, "__name__", "") != "_collect_jax_warning":
    warnings.showwarning = _collect_jax_warning
