# What a uniform draw returns

Related: [rng-configuration.md](rng-configuration.md) · [nn-library.md](nn-library.md) · [core-types.md](core-types.md)

Every uniform random value in Shorokoo comes out of one draw: `Globals.RandomUniform`, the
`Uniform` / `UniformRange` / `XavierUniform` / `KaimingUniform` / `RecurrentUniform` /
`XavierUniformGain` / `KaimingUniformGain` initializers, and the mask a `Dropout` layer
builds. This page is that draw's contract — the interval, the edge cases, how finely it
resolves a range, and the places where it is not perfect. For *which* values a given draw
produces (seeds, streams, reproducibility) see
[Configuring randomness](rng-configuration.md).

## Facts

- The interval is **half-open**: a draw over a non-empty `[low, high)` returns values
  `>= low` and `< high`, and never `high` — matching PyTorch's `uniform_`, Keras's
  `RandomUniform`, and ONNX's `RandomUniform`. A degenerate range is the exception: when
  `low == high` there is no interval to draw from and every element is that bound, which
  is `high`'s value as much as it is `low`'s (see [the table below](#degenerate-and-non-finite-bounds)).
- The result dtype is always **`float32`**, whatever the bounds are.
- The draw is **uniform in value**: the chance of landing in a sub-interval is proportional
  to that sub-interval's width, for any range you ask for — exactly so, but for the bounded
  imperfections at the bottom of this page.
- Bounds may be compile-time literals or graph scalars computed in-graph — the two
  `RandomUniform` overloads in [core-types.md](core-types.md#factory-helpers-using-static-shorokooglobals),
  and every initializer that takes a bound as an `Init` argument. Both forms use the same
  draw and carry the same guarantees (graph-scalar bounds additionally need a keyed —
  concrete, id-bearing — model, since they cannot be expressed as ONNX attributes).
- Nothing here rests on floating-point rounding: the draw is integer and exact, so the same
  seed yields identical values under ONNX Runtime's CPU provider, under the Quick Execution
  Engine, and in an exported ONNX model — the three paths the suite compares bit for bit.
  Other execution providers are expected to agree for the same reason, with one risk still
  open: a provider that flushes subnormals to zero would disturb the sub-floor lattice
  described below ([issue #160](https://github.com/Shorokoo/Shorokoo/issues/160)).

## The range is addressed, not scaled

A uniform over an arbitrary range is commonly built by drawing `u` on `[0, 1)` and
returning `u·(high − low) + low`. Shorokoo does not do that: it addresses the `float32`
values of the range directly, weighting each one by the width of the real interval it
stands for. Three guarantees follow.

- **No precision is lost near zero.** Over `[-1, 1)` the draw resolves magnitudes down to
  2⁻⁴⁰; scaling a standard draw would round every result near zero to a multiple of about
  2⁻²³, so the small values a symmetric initializer is supposed to produce mostly collapse
  onto a coarse grid.
- **A range wider than `float32` does not overflow.**
  `UniformRange.Init([shape], Scalar(-1.8e38f), Scalar(1.8e38f))` draws normally, where
  `high − low` alone would be `+infinity`.
- **`high` is never returned from a non-empty range.** The exclusion is structural —
  `high`'s own float is not one of the values the draw can address — rather than a property
  that a rounding step could undo. Only the degenerate `low == high`, which addresses no
  float at all and fills with the bound, gives that value back.

## Degenerate and non-finite bounds

A uniform draw never throws on its bounds. It resolves them like this:

| Bounds | Result |
|---|---|
| `low == high` | that value, for every element (`-0f` normalises to `+0f`) |
| `low > high` | `low`, for every element (`-0f` normalises to `+0f`) |
| `low` is NaN | that NaN, sign and payload intact |
| `high` is NaN | that NaN, sign and payload intact |
| both bounds NaN | `low`'s NaN |
| `low = float.NegativeInfinity` | behaves as `-float.MaxValue` |
| `high = float.PositiveInfinity` | every finite float above `low` stays drawable, `float.MaxValue` included |
| `low = float.PositiveInfinity` | behaves as `float.MaxValue`, which leaves any finite `high` inverted — so you get `float.MaxValue` |
| `high = float.NegativeInfinity` | behaves as `-float.MaxValue`, inverted for any larger `low` — so you get `low` |

So `RandomUniform(shape, float.NegativeInfinity, float.PositiveInfinity)` draws over the
whole finite `float32` domain and never returns an infinity or a NaN — `+infinity` as the
upper bound is the one non-finite bound that widens the range instead of clamping it.

The initializers that take bounds (`UniformRange` and friends) document `low <= high` as
their expectation; an inverted range is not an error, it is a constant fill with `low`.

## How finely the range is resolved

The draw resolves **41 successive weight classes** — a class is `max(1, exponent field)`,
which is a binade of magnitude everywhere except at the very bottom, where the subnormals
and the smallest normal binade fold into one class — counting down from the largest
magnitude in the range, or **40** when the range straddles zero. Call the bottom of that
span the *floor*:

- **Above the floor**, every single `float32` value in the range is drawable, with
  probability proportional to its ulp: exactly so when the range's total weight is a power
  of two, and otherwise to within one weight unit, the smallest share the draw can express.
- **Below the floor**, values come from an evenly spaced lattice whose step is 2⁻²³ of the
  floor, so those floats are *represented* but not individually addressable: the draw can
  land in that region and will land there as often as its width says it should, to within
  the same one weight unit where the range's end cuts a lattice cell short, but it cannot
  single out an arbitrary float down there.

On `[0, 1)` the floor is 2⁻⁴¹: every float from 2⁻⁴¹ up is individually drawable, and
smaller results are multiples of 2⁻⁶⁴ (exact `0f` among them). On `[-1, 1)` the floor is
2⁻⁴⁰ — straddling zero costs one class, since both signs of every magnitude are in play.

Counting values rather than mass: about **33.1%** of the floats in `[0, 1)` can come out of
a draw over `[0, 1)`, and about **16.1%** of the floats in the whole finite `float32`
domain can come out of a draw over that domain. What truncation does *not* spend is the
probability mass — each region keeps the share its width earns, exactly when the range's
total weight is a power of two and otherwise to within one weight unit, so the draw stays
uniform in value either way. Resolution is what truncation spends, not fairness.

For practical ranges this is invisible: an initializer bound, a `[0, 1)` mask, a
`[-a, a)` weight draw all live far above their floor. It becomes observable when a range
spans dozens of orders of magnitude at once.

## Known imperfections

Three, none of them large, all of them real:

- **A single float can take up to twice its due share.** Whether the draw divides evenly
  depends on the range: when its total weight is a power of two the split is *exact*, and
  that covers the common cases — `[0, 1)`, `[-1, 1)`, `[4, 12)`, `[0, +infinity)`. Otherwise
  the lightest floats round up or down by one draw in 2⁶⁴, which for a float carrying the
  smallest possible weight is a factor of at most 2 against its neighbour. The error does
  not accumulate: over any run of adjacent floats the count stays within one draw of due,
  so the skew is visible only per-float and only at the very lightest of them.
- **`low` itself is not always drawable.** Where `low` sits above the floor it is drawable
  and carries exactly one float's share. Where it falls below the floor and off the lattice
  it cannot be returned at all — e.g. a draw over `[-1, 1e30)` never returns exactly `-1`.
- **A vanishingly small side of a hugely lopsided range can get probability exactly zero.**
  A draw over `[-1, 1e30)` returns no negative value at all: the negative side is worth
  about 10⁻³⁰ of the range, which is less than the smallest share the draw can express, and
  it is dropped rather than rounded up to the smallest expressible one — which would
  over-weight that sliver by far more than dropping it costs. If you need both signs
  represented, do not pair bounds whose magnitudes differ by 30-odd orders of magnitude.

## Negative zero is never returned

A draw never returns `-0f`, whatever you pass. `-0f` can only enter through `low` — a drawn
value is never a negative zero, and `high` is an exclusive bound — and a `low` of `-0f` is
normalised to `+0f` before anything else looks at it. So `RandomUniform(shape, -0f, 0f)`,
`RandomUniform(shape, -0f, -0f)` and an inverted range starting at `-0f` all yield `+0f`
rather than the `-0f` the "returns `low`" rule above would otherwise give. The two values
are numerically equal, so only a bit comparison can see the difference — but the guarantee
is a bit-level one, and it is enforced on the bounds before any drawing happens, so it does
not depend on how an execution provider treats a negative zero. It is checked bit for bit
under ONNX Runtime's CPU provider, under the Quick Execution Engine, and on an exported
ONNX model.
