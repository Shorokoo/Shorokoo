"""ONNX strings and text: StringConcat, StringSplit, StringNormalizer, RegexFullMatch,
TfIdfVectorizer.

String tensors are numpy object arrays on the host (torch has no strings); a result that is not
text -- a match, a count, a frequency -- is a torch tensor on the run's device.
"""

import numpy as np
import torch

from . import runtime as _rt


def _on_device(array, dtype):
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype).to(_rt.device())


def string_concat(x, y):
    return np.asarray(np.add(x.astype(object), y.astype(object)), dtype=object)


def string_split(x, /, *, delimiter=None, maxsplit=None):
    """StringSplit: each string split at `delimiter` -- at runs of whitespace, ignoring any at the
    ends, when there is none -- at most `maxsplit` times; the pieces fill a new last axis padded
    with empty strings, and the count of each string's pieces is the second output."""
    separator = delimiter if delimiter else None
    limit = -1 if maxsplit is None else int(maxsplit)
    pieces = [s.split(separator, limit) for s in x.reshape(-1).tolist()]
    width = max((len(p) for p in pieces), default=0)
    y = np.full((len(pieces), width), "", dtype=object)
    for row, parts in enumerate(pieces):
        y[row, :len(parts)] = parts
    counts = np.array([len(p) for p in pieces], dtype=np.int64).reshape(x.shape)
    return y.reshape(list(x.shape) + [width]), _on_device(counts, torch.int64)


def string_normalizer(x, /, *, case_change_action="NONE", is_case_sensitive=0, locale="", stopwords=None):
    """StringNormalizer over a [C] or [1, C] tensor: the stopwords removed (compared ignoring case
    unless `is_case_sensitive`), then the case changed. Where nothing is left the result is one
    empty string."""
    values = x.reshape(-1).tolist()
    if stopwords:
        if is_case_sensitive:
            stop = set(stopwords)
            values = [v for v in values if v not in stop]
        else:
            stop = {w.lower() for w in stopwords}
            values = [v for v in values if v.lower() not in stop]
    if case_change_action == "LOWER":
        values = [v.lower() for v in values]
    elif case_change_action == "UPPER":
        values = [v.upper() for v in values]
    if not values:
        values = [""]
    shape = [len(values)] if x.ndim == 1 else [1, len(values)]
    return _rt.strings(values, shape)


def regex_full_match(x, /, *, pattern):
    """RegexFullMatch in RE2's syntax, as ONNX specifies: its \\d, \\w and \\s are ASCII, it has
    POSIX classes, \\pL and \\x{41}, and neither backreferences nor lookarounds."""
    import re2

    matcher = re2.compile(pattern)
    found = np.array([matcher.fullmatch(s) is not None for s in x.reshape(-1).tolist()], dtype=bool)
    return _on_device(found.reshape(x.shape), torch.bool)


def tf_idf_vectorizer(x, /, *, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts,
                      ngram_indexes, pool_int64s=None, pool_strings=None, weights=None):
    """TfIdfVectorizer: counts of the pool's n-grams in each row of `x` ([C] or [N, C]), n-grams
    being taken with every skip up to `max_skip_count` (unigrams once), weighted as `mode` says, at
    the output positions `ngram_indexes` gives them."""
    pool = list(pool_strings) if pool_strings is not None else [int(v) for v in pool_int64s]
    grams = {}
    for n, start in enumerate(ngram_counts, start=1):
        end = ngram_counts[n] if n < len(ngram_counts) else len(pool)
        for offset in range(start, end, n):
            grams[tuple(pool[offset:offset + n])] = len(grams)
    prefixes = {gram[:i] for gram in grams for i in range(1, len(gram) + 1)}

    host = x if _rt.is_strings(x) else x.detach().cpu().numpy()
    rows = host.reshape(1, -1) if host.ndim <= 1 else host
    items = [[v if isinstance(v, str) else int(v) for v in row.tolist()] for row in rows]
    width = max(ngram_indexes) + 1 if len(ngram_indexes) else 0
    counts = np.zeros((len(items), width), dtype=np.float32)
    for r, row in enumerate(items):
        first = min_gram_length
        for skip in range(1, max_skip_count + 2):
            for start in range(len(row)):
                if start + skip * (first - 1) >= len(row):
                    break
                gram = ()
                position = start
                while len(gram) < max_gram_length and position < len(row):
                    gram = gram + (row[position],)
                    if gram not in prefixes:
                        break
                    if len(gram) >= first and gram in grams:
                        counts[r, ngram_indexes[grams[gram]]] += 1
                    position += skip
            if first == 1:
                first = 2
                if first > max_gram_length:
                    break

    if mode == "IDF":
        counts = (counts > 0).astype(np.float32)
    if mode in ("IDF", "TFIDF") and weights is not None and len(weights):
        # A weight per output position, as ONNX Runtime and the reference implementation read them.
        scale = np.zeros(width, dtype=np.float32)
        known = min(width, len(weights))
        scale[:known] = weights[:known]
        counts = counts * scale
    result = counts.reshape(-1) if host.ndim <= 1 else counts
    return _on_device(result, torch.float32)
