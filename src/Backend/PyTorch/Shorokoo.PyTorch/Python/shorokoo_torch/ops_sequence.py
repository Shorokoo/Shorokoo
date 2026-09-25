"""ONNX sequences and optionals: SequenceConstruct, SequenceEmpty, SequenceInsert, SequenceErase,
SequenceAt, SequenceLength, SplitToSequence, ConcatFromSequence, SequenceMap, Optional,
OptionalHasElement, OptionalGetElement.

A sequence is a Python list and every operator here makes a new one, so a sequence a node was
handed is never changed under another node that reads it. An optional is its value, or None when
it has none.
"""

import numpy as np
import torch

from . import runtime as _rt


def _position(position, length, default):
    if position is None:
        return default
    index = int(position.reshape(-1)[0].item())
    return index + length if index < 0 else index


def sequence_construct(*inputs):
    return list(inputs)


def sequence_empty(*, dtype=None):
    return []


def sequence_insert(sequence, tensor, position=None, /):
    result = list(sequence)
    result.insert(_position(position, len(sequence), len(sequence)), tensor)
    return result


def sequence_erase(sequence, position=None, /):
    result = list(sequence)
    del result[_position(position, len(sequence), len(sequence) - 1)]
    return result


def sequence_at(sequence, position):
    return sequence[_position(position, len(sequence), None)]


def sequence_length(sequence):
    return torch.tensor(len(sequence), dtype=torch.int64, device=_rt.device())


def split_to_sequence(x, split=None, /, *, axis=0, keepdims=1):
    """SplitToSequence: chunks of 1 along `axis` without `split` (the axis dropped unless
    `keepdims`), of `split`'s size when it is a scalar (the last one shorter), or of its sizes when
    it is a vector. `keepdims` is ignored when `split` is given, as the spec says."""
    length = x.shape[axis]
    if split is None:
        sizes = [1] * length
    elif split.ndim == 0:
        size = int(split.item())
        sizes = [size] * (length // size) + ([length % size] if length % size else [])
    else:
        sizes = [int(v) for v in split.reshape(-1).tolist()]
    if _rt.is_strings(x):
        chunks = np.split(x, np.cumsum(sizes)[:-1], axis=axis)
    else:
        chunks = list(torch.split(x, sizes, axis))
    if split is None and not keepdims:
        chunks = [chunk.squeeze(axis) if not _rt.is_strings(chunk) else np.squeeze(chunk, axis) for chunk in chunks]
    return chunks


def concat_from_sequence(sequence, /, *, axis, new_axis=0):
    if _rt.is_strings(sequence[0]):
        return np.stack(sequence, axis=axis) if new_axis else np.concatenate(sequence, axis=axis)
    return torch.stack(sequence, axis) if new_axis else torch.cat(sequence, axis)


def sequence_map(body, sequence, *additional, _outputs):
    """SequenceMap: `body` run once per element of `sequence`, each additional input passed whole
    or, where it is a sequence, element by element; each of the body's `_outputs` outputs is
    gathered into a sequence of its own."""
    results = [[] for _ in range(_outputs)]
    for i in range(len(sequence)):
        outputs = body(sequence[i], *[a[i] if isinstance(a, list) else a for a in additional])
        for gathered, value in zip(results, outputs):
            gathered.append(value)
    return tuple(results)


def optional(value=None, /):
    return value


def optional_has_element(value=None, /):
    return torch.tensor(value is not None, device=_rt.device())


def optional_get_element(value, /):
    if value is None:
        raise ValueError("OptionalGetElement: the optional has no value")
    return value
