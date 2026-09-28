# What a uniform draw returns

Related: [rng-configuration.md](rng-configuration.md) · [nn-library.md](nn-library.md) · [core-types.md](core-types.md) · [limitations.md](limitations.md)

Every uniform random value in Shorokoo comes out of one draw: `Globals.RandomUniform`, the
`Uniform` / `UniformRange` / `XavierUniform` / `KaimingUniform` / `RecurrentUniform` /
`XavierUniformGain` / `KaimingUniformGain` initializers, and the mask a `Dropout` layer
builds. This page is that draw's contract: the interval, the edge cases, how finely it
resolves a range, and its imperfections. For *which* values a given draw produces (seeds,
streams, reproducibility) see [Configuring randomness](rng-configuration.md).

```csharp
var mask   = RandomUniform(Vector(4L, 8L));                  // float32 in [0, 1)
var weight = RandomUniform(Vector(4L, 8L), -0.05f, 0.05f);   // float32 in [-0.05, 0.05)
var init   = UniformRange.Init([Scalar(4L), Scalar(8L)], Scalar(-1f), Scalar(1f));  // in-graph bounds
```

## Facts

- The interval is **half-open**: a draw over a non-empty `[low, high)` returns values
  `>= low` and `< high`, never `high`, matching PyTorch's `uniform_`, Keras's
  `RandomUniform` and ONNX's `RandomUniform`. When `low == high`, every element is that
  bound (see [the table below](#degenerate-and-non-finite-bounds)).
- The result dtype is always **`float32`**, whatever the bounds are.
- The draw is **uniform in value**: the chance of landing in a sub-interval is proportional
  to its width, up to the bounded [imperfections](#known-imperfections). Per float, that
  means **pick a real in `[low, high)` and round down**, which is *not* what
  `low + (high − low)·u` gives; see [the next section](#the-range-is-addressed-not-scaled).
- Bounds may be compile-time literals or graph scalars computed in-graph: the two
  `RandomUniform` overloads in [core-types.md](core-types.md#factory-helpers-using-static-shorokooglobals),
  and every [initializer](nn-library.md#initializers-shorokoomodulesinitializers) that takes a
  bound as an `Init` argument. Both forms use the same draw and carry the same guarantees.
  Graph-scalar bounds cannot be expressed as ONNX attributes, so they also need a
  concrete model built through [`ToConcreteModel`](rng-configuration.md), not a bare
  architecture.
- The draw is integer and exact, so the same seed yields identical values bit for bit
  under ONNX Runtime's CPU provider, under the
  [Quick Execution Engine](limitations.md#quick-execution-engine-value-computation-is-bounded),
  and in an exported ONNX model. Other execution providers are expected to agree for the
  same reason, with one untested risk: a provider that flushes very small magnitudes to
  zero would disturb the values a draw produces closest to zero
  ([issue #160](https://github.com/Shorokoo/Shorokoo/issues/160)).

## The range is addressed, not scaled

The distribution: **pick a real number uniformly from `[low, high)` and round it down to a
`float32`.** Equivalently, each float comes out with probability proportional to its
**ulp** (the width of the real interval it stands for), so where floats are dense each is
correspondingly rarer. Two bounded qualifications apply, set out under
[how finely the range is resolved](#how-finely-the-range-is-resolved): the rounding lands on
a `float32` only down to a floor, beneath which it lands on a coarser grid; and the shares
are exact for a range whose width is a power of two, and otherwise off by at most a factor
of two, only on the lightest floats.

Most standard libraries instead draw `u` on `[0, 1)` and return `u·(high − low) + low`,
which inherits `u`'s granularity and lands on a coarse grid wherever the range's floats are
finer. Shorokoo addresses the `float32` values of the range directly, which gives three
guarantees.

- **No precision is lost near zero.** Over `[-1, 1)` the draw resolves magnitudes down to
  2⁻⁴⁰; scaling a standard draw would round every result near zero to a multiple of about
  2⁻²³, collapsing the small values a symmetric initializer produces onto a coarse grid.
- **A range wider than `float32` does not overflow.**
  `UniformRange.Init([shape], Scalar(-1.8e38f), Scalar(1.8e38f))` draws normally, where
  `high − low` alone would be `+infinity`.
- **`high` is never returned from a non-empty range.** `high`'s float is not one of the
  values the draw can address, so no rounding step can produce it. Only the degenerate
  `low == high`, which fills with the bound, gives that value back.

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
| `high = float.PositiveInfinity` | the range runs to every finite float above `low`, `float.MaxValue` included |
| `low = float.PositiveInfinity` | behaves as `float.MaxValue`, which leaves any finite `high` inverted — so you get `float.MaxValue` |
| `high = float.NegativeInfinity` | behaves as `-float.MaxValue`, inverted for any larger `low` — so you get `low` |

So `RandomUniform(shape, float.NegativeInfinity, float.PositiveInfinity)` draws over the
whole finite `float32` domain and never returns an infinity or a NaN; `+infinity` as the
upper bound is the one non-finite bound that widens the range instead of clamping it.

The initializers that take bounds (`UniformRange` and friends) expect `low <= high`; an
inverted range is a constant fill with `low`, not an error.

## How finely the range is resolved

The draw resolves **41 successive weight classes**, counting down from the largest magnitude
in the range — **40** when the range straddles zero, since both signs of every magnitude are
then in play. Call the bottom of that span the *floor*.

A **weight class** is one power of two of magnitude: 2²³ `float32` values that share one
ulp. The *subnormals* (the values below the smallest normal magnitude) are the exception:
they share a class with the smallest normal span.

A float's **weight** is its ulp in units of the ulp at the floor, so a float one class above
the floor weighs 2, two classes above weighs 4, and so on. That doubling fixes the depth at
41: 41 classes of 2²³ floats weighing 2ᵏ each come to 2²³·(2⁴¹ − 1), and the sub-floor
lattice's 2²³ points bring it to exactly 2⁶⁴, the number of values one 64-bit generator draw
can take. A 42nd class does not fit.

A range's **total weight** is its width in those units, rounded down:
`floor((high − low) / (the ulp at the floor))`. The rounding drops under one unit, the same
sliver the lattice loses where the range's end cuts a cell short. Since the ulp at the floor
is a power of two, **a power-of-two width divides cleanly** and the total is a power of two
(`[0, 1)`, `[-1, 1)`, `[4, 12)`, `[0, +infinity)`, but not an arbitrary `[-a, a)` bound).
Handing the 2⁶⁴ draws out over that total is then exact; otherwise each weight unit gets
one draw more or less than its due.

The floor divides two regimes:

- **Above the floor**, every single `float32` value in the range is drawable, with
  probability proportional to its ulp.
- **Below the floor**, values come from an evenly spaced lattice whose step is 2⁻²³ of the
  floor. The draw lands in that region as often as its width says, to within one weight unit
  where the range's end cuts a lattice cell short, but cannot single out an arbitrary float
  there.

On `[0, 1)` the floor is 2⁻⁴¹: every float from 2⁻⁴¹ up is individually drawable, and
smaller results are multiples of 2⁻⁶⁴, exact `0f` among them. Above that floor this is bit
for bit the classical dense construction for the unit interval (Walker's 1974 method, in
the 41-plus-23-bit form Marc Reynolds gives it), so `[0, 1)` costs nothing against a
generator built for that range alone. On `[-1, 1)` the floor is 2⁻⁴⁰.

Counting values: about **33.1%** of the floats in `[0, 1)` can come out of a draw over
`[0, 1)`, and about **16.1%** of the floats in the whole finite `float32` domain can come
out of a draw over that domain. Truncation spends resolution, not probability mass: every
region keeps the share its width earns, up to the same rounding, so the draw stays uniform
in value.

The collapsed floats carry almost no mass. Wherever the lattice costs resolution, a draw
reaches it on the order of **once in a trillion** times: 2⁻⁴¹, or 4.5e-13, over `[0, 1)`,
`[0, float.MaxValue)` and `[0, +infinity)`, rising to 2.4e-12 (about one draw in 4e11) for a
range that straddles zero, which puts both signs of the collapsed span on the lattice and
doubles its weight. `[-0.1, 0.3)` sits near that ceiling as squarely as `[-1e30, 2e18)`
does: the straddle costs, not the spread.

When a range is small enough that its 41 classes (40 if it straddles zero) reach the bottom
of the format, the lattice step equals the spacing of the float grid, so every lattice point
is an exactly addressed float and nothing is collapsed.

For practical ranges (an initializer bound, a `[0, 1)` mask, a `[-a, a)` weight draw) this
is invisible: they live far above their floor. It becomes observable when a range spans
dozens of orders of magnitude.

## Known imperfections

All three follow from spending one 64-bit generator value per element: the smallest share
the draw can express is one part in 2⁶⁴, and the truncation depth is set by the same budget.

- **A single float can take up to twice its due share.** A power-of-two width divides
  exactly (see [above](#how-finely-the-range-is-resolved)); a bound like `√(6 / fanIn)` does
  not. Then every weight unit rounds up or down by one draw in 2⁶⁴, which leaves any float
  within a factor of `(q+1)/q` of its due share: at most 2, reached only by the lightest
  floats, which have the least weight to absorb the rounding. The factor does *not* bound
  the error in draws: a heavy float's weight units are spread across the axis, so their
  roundings add up instead of cancelling, and a heavy float or a run of neighbours drifts
  further in absolute terms than a light one. The share stays within the factor.
- **`low` itself is not always drawable.** Where `low` sits above the floor it is drawable
  and carries exactly one float's share. Where it falls below the floor and off the lattice
  it cannot be returned at all — e.g. a draw over `[-1, 1e30)` never returns exactly `-1`.
- **A vanishingly small side of a hugely lopsided range can get probability exactly zero.**
  A draw over `[-1, 1e30)` returns no negative value at all: the negative side is worth
  about 10⁻³⁰ of the range, less than the smallest share the draw can express, and is
  dropped rather than rounded up. If you need both signs represented, do not pair bounds
  whose magnitudes differ by 30-odd orders of magnitude.

## Negative zero is never returned

A draw never returns `-0f`, whatever you pass. `-0f` can only enter through `low` (a drawn
value is never a negative zero, and `high` is exclusive), and a `low` of `-0f` is normalised
to `+0f` first. So `RandomUniform(shape, -0f, 0f)`, `RandomUniform(shape, -0f, -0f)` and an
inverted range starting at `-0f` all yield `+0f`, not the `-0f` the "returns `low`" rule
would give. The guarantee is bit-level and enforced on the bounds before drawing, so it does
not depend on how an execution provider treats a negative zero.
