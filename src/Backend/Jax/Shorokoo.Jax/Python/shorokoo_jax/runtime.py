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

# Products and convolutions in full float32 precision: XLA's default on a card may round the
# operands of a float32 product to TensorFloat-32, and the backend computes what the graph says.
PRECISION = jax.lax.Precision.HIGHEST


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


def empty(code, shape, device_name):
    """A tensor whose contents are unspecified: zeros, since a jax array is never uninitialized."""
    if device_name == "cpu":
        return np.zeros(tuple(shape), dtype=jax_dtype(code))
    return jax.device_put(np.zeros(tuple(shape), dtype=jax_dtype(code)), device_of(device_name))


def host_copy(value):
    """A host value of its own with `value`'s contents: C-contiguous and writable."""
    array = np.array(value, copy=True, order="C")
    return array if array.flags.writeable else array.copy()


def describe(value):
    """(kind, element type code, shape, is host, data address, byte count) of a value."""
    if isinstance(value, np.ndarray):
        if not value.flags.c_contiguous or not value.flags.writeable:
            # The .NET side reads and writes a host value as one dense buffer from its address.
            raise ValueError("only a contiguous, writable array can be handed to .NET")
        return (KIND_TENSOR, dtype_code(value), list(value.shape), True, value.ctypes.data, value.nbytes)
    return (KIND_TENSOR, dtype_code(value), list(value.shape), False, 0,
            int(np.prod(value.shape, dtype=np.int64)) * np.dtype(value.dtype).itemsize)


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

    def __init__(self, main, constants, device):
        self._main = main
        self._constants = constants
        self.device = device
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
        try:
            with np.errstate(all="ignore"):
                return tuple(self._main(*args))
        finally:
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


def load_model(source, filename, constants, device_name):
    """The model a translation's source defines, with `constants` bound as `_C`, on `device_name`.
    The source's compiled Python code is cached by `filename`, which names the source's hash."""
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
    return Model(namespace["main"], constants, device_of(device_name))


def prepare(model, inputs, severity=None):
    token = _warning_severity.set(severity)
    try:
        model.prepare(inputs)
    finally:
        _warning_severity.reset(token)


def run(model, args, wanted, retained, severity=None):
    """Runs a model and hands over the outputs at indices `wanted`: each retained in the device's
    memory where `retained` says so, else copied home. Returns (value, description) per output.
    Every output is ready when this returns: nothing the run reads or writes is still in flight."""
    token = _warning_severity.set(severity)
    try:
        outputs = model(args)
    finally:
        _warning_severity.reset(token)
    results = []
    for index, retain in zip(wanted, retained):
        value = outputs[index]
        value = jax.block_until_ready(value) if retain else host_copy(value)
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
    shrinkages (0), reserves (0), held in all. None where the device keeps none."""
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

WARNING = 2
_warning_severity = contextvars.ContextVar("shorokoo_jax_warning_severity", default=None)
_show_warning = warnings.showwarning


def _show_warning_at_jax_severity(message, category, filename, lineno, file=None, line=None):
    """Shows a warning unless the run that raised it asked only for errors: the session's log
    severity, read per run, since the interpreter's warning filters are the whole process's. It
    hands on to whatever showed warnings before, another backend's filter included."""
    severity = _warning_severity.get()
    if severity is not None and severity > WARNING:
        return
    _show_warning(message, category, filename, lineno, file, line)


if getattr(warnings.showwarning, "__name__", "") != "_show_warning_at_jax_severity":
    warnings.showwarning = _show_warning_at_jax_severity
