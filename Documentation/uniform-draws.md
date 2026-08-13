# What a uniform draw returns

Related: [rng-configuration.md](rng-configuration.md) · [nn-library.md](nn-library.md) · [core-types.md](core-types.md) · [limitations.md](limitations.md)

Every uniform random value in Shorokoo comes out of one draw: `Globals.RandomUniform`, the
`Uniform` / `UniformRange` / `XavierUniform` / `KaimingUniform` / `RecurrentUniform` /
`XavierUniformGain` / `KaimingUniformGain` initializers, and the mask a `Dropout` layer
builds. This page is that draw's contract — the interval, the edge cases, how finely it
resolves a range, and the places where it is not perfect. For *which* values a given draw
produces (seeds, streams, reproducibility) see
[Configuring randomness](rng-configuration.md).

```csharp
var mask   = RandomUniform(Vector(4L, 8L));                  // float32 in [0, 1)
var weight = RandomUniform(Vector(4L, 8L), -0.05f, 0.05f);   // float32 in [-0.05, 0.05)
var init   = UniformRange.Init([4L, 8L], Scalar(-1f), Scalar(1f));   // same draw, in-graph bounds
```

## Facts

- The interval is **half-open**: a draw over a non-empty `[low, high)` returns values
  `>= low` and `< high`, and never `high` — matching PyTorch's `uniform_`, Keras's
  `RandomUniform`, and ONNX's `RandomUniform`. A degenerate range is the exception: when
  `low == high` there is no interval to draw from and every element is that bound, which
  is `high`'s value as much as it is `low`'s (see [the table below](#degenerate-and-non-finite-bounds)).
- The result dtype is always **`float32`**, whatever the bounds are.
- The draw is **uniform in value**: the chance of landing in a sub-interval is proportional
  to that sub-interval's width, but for the bounded imperfections at the bottom of this page.
  Per float, that means **pick a real in `[low, high)` and round down**, which is *not* what
  `low + (high − low)·u` gives — see [the next section](#the-range-is-addressed-not-scaled).
- Bounds may be compile-time literals or graph scalars computed in-graph — the two
  `RandomUniform` overloads in [core-types.md](core-types.md#factory-helpers-using-static-shorokooglobals),
  and every [initializer](nn-library.md#initializers-shorokoomodulesinitializers) that takes a
  bound as an `Init` argument. Both forms use the same draw and carry the same guarantees;
  graph-scalar bounds cannot be expressed as ONNX attributes, so they additionally need a
  concrete model — one built through
  [`ToConcreteModel`](rng-configuration.md), not a bare architecture.
- Nothing here rests on floating-point rounding: the draw is integer and exact, so the same
  seed yields identical values bit for bit under ONNX Runtime's CPU provider, under the
  [Quick Execution Engine](limitations.md#quick-execution-engine-value-computation-is-bounded),
  and in an exported ONNX model. Other execution providers are expected to agree for the
  same reason, with one risk untested for want of the hardware: a provider that flushes very
  small magnitudes to zero would disturb the values a draw produces closest to zero
  ([issue #160](https://github.com/Shorokoo/Shorokoo/issues/160)).

## The range is addressed, not scaled

One sentence covers the distribution: **pick a real number uniformly from `[low, high)` and
round it down to a `float32`.** Equivalently, each float comes out with probability
proportional to its **ulp** — the width of the real interval it stands for — so wherever
floats are dense each individual one is correspondingly rarer, and every sub-interval takes
the share its width earns. Two bounded qualifications apply, both set out under
[how finely the range is resolved](#how-finely-the-range-is-resolved): the rounding lands on
a `float32` only down to a floor, beneath which it lands on a coarser grid; and the shares
are exact for a range whose width is a power of two, and for any other range are off by at
most a factor of two, and that only on the very lightest floats.

That is not the usual construction. A uniform over an arbitrary range is commonly built — as
most standard libraries build it — by drawing `u` on `[0, 1)` and returning
`u·(high − low) + low`, which inherits `u`'s own granularity and so lands on a coarse grid
wherever the range's floats are finer than that. Shorokoo addresses the `float32` values of
the range directly instead, and three guarantees follow.

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
| `high = float.PositiveInfinity` | the range runs to every finite float above `low`, `float.MaxValue` included |
| `low = float.PositiveInfinity` | behaves as `float.MaxValue`, which leaves any finite `high` inverted — so you get `float.MaxValue` |
| `high = float.NegativeInfinity` | behaves as `-float.MaxValue`, inverted for any larger `low` — so you get `low` |

So `RandomUniform(shape, float.NegativeInfinity, float.PositiveInfinity)` draws over the
whole finite `float32` domain and never returns an infinity or a NaN — `+infinity` as the
upper bound is the one non-finite bound that widens the range instead of clamping it.

The initializers that take bounds (`UniformRange` and friends) expect `low <= high`; an
inverted range is not an error, it is a constant fill with `low`.

## How finely the range is resolved

The draw resolves **41 successive weight classes**, counting down from the largest magnitude
in the range — **40** when the range straddles zero, since both signs of every magnitude are
then in play. Call the bottom of that span the *floor*.

A **weight class** is one power of two of magnitude: 2²³ `float32` values that all share one
ulp. The very bottom of the format is the one exception — the *subnormals*, the tiny values
below the smallest normal magnitude, share a class with the smallest normal span rather than
forming classes of their own.

**Weight** is how the draw counts shares. A float's weight is its ulp measured in units of
the smallest ulp resolved — the ulp at the floor — so a float one class above the floor
weighs 2, two classes above weighs 4, and so on up. That doubling is what fixes the depth at
41: a class of 2²³ floats weighing 2ᵏ each, summed over 41 classes, comes to 2²³·(2⁴¹ − 1) —
just under 2⁶⁴, the number of values one 64-bit generator draw can take. A 42nd class would
overrun it.

A range's **total weight** is simply its width in those same units:
`(high − low) / (the ulp at the floor)`. Since the ulp at the floor is itself a power of two,
the total is a power of two exactly when **the range's width is** — which is what `[0, 1)`,
`[-1, 1)`, `[4, 12)` and `[0, +infinity)` have in common, and what an arbitrary `[-a, a)`
initializer bound does not. Handing the 2⁶⁴ draws out over that total is exact in the first
case, and in the second leaves each weight unit one draw above or below its due.

The floor divides two regimes:

- **Above the floor**, every single `float32` value in the range is drawable, with
  probability proportional to its ulp.
- **Below the floor**, values come from an evenly spaced lattice whose step is 2⁻²³ of the
  floor, so those floats are *represented* but not individually addressable: the draw can
  land in that region and will land there as often as its width says it should, to within
  one weight unit where the range's end cuts a lattice cell short, but it cannot single out
  an arbitrary float down there.

On `[0, 1)` the floor is 2⁻⁴¹: every float from 2⁻⁴¹ up is individually drawable, and
smaller results are multiples of 2⁻⁶⁴, exact `0f` among them. Above that floor this is bit
for bit the classical dense construction for the unit interval — Walker's 1974 method, in
the 41-plus-23-bit form Marc Reynolds gives it — so asking for `[0, 1)` through the
arbitrary-range machinery costs nothing against a generator built for that range alone. On
`[-1, 1)` the floor is 2⁻⁴⁰.

Counting values rather than mass: about **33.1%** of the floats in `[0, 1)` can come out of
a draw over `[0, 1)`, and about **16.1%** of the floats in the whole finite `float32`
domain can come out of a draw over that domain. What truncation does *not* spend is the
probability mass: every region keeps the share its width earns, up to that same rounding, so
the draw stays uniform in value either way. Resolution is what truncation spends, not
fairness.

The floats truncation collapses are also the ones carrying almost no mass. Wherever the
lattice costs resolution at all, a draw reaches it on the order of **once in a trillion**
times: 2⁻⁴¹, or 4.5e-13, over `[0, 1)`, `[0, float.MaxValue)` and `[0, +infinity)`, rising to
2.4e-12 — about one draw in 4e11 — for a range that straddles zero, which puts both signs of
the collapsed span on the lattice and so doubles its weight. `[-0.1, 0.3)` sits near that
ceiling as squarely as `[-1e30, 2e18)` does; it is the straddle that costs, not the spread.
A large fraction of the *floats*, a negligible fraction of the *draws*.

A range small enough that its 41 classes — 40 if it straddles zero — reach the bottom of the
format is the exception, and a harmless one. There the lattice step *is* the spacing of the
float grid, so every point of the lattice is an exactly addressed float: nothing is
collapsed, and a draw that lands below the floor loses nothing by it.

For practical ranges this is invisible: an initializer bound, a `[0, 1)` mask, a
`[-a, a)` weight draw all live far above their floor. It becomes observable when a range
spans dozens of orders of magnitude at once.

## Known imperfections

All three follow from spending exactly one 64-bit generator value per element: the smallest
share the draw can express is one part in 2⁶⁴, and the truncation depth is set by the same
budget.

- **A single float can take up to twice its due share.** Whether the draw divides evenly
  depends on the range: a power-of-two width divides exactly, so `[0, 1)`, `[-1, 1)`,
  `[4, 12)` and `[0, +infinity)` are clean, while a bound like `√(6 / fanIn)` is not.
  Otherwise the lightest floats round up or down by one draw in 2⁶⁴, which for a float
  carrying the
  smallest possible weight is a factor of at most 2 against its neighbour. The error does
  not accumulate: a run of adjacent floats is no further off than a single one, so the skew
  is visible only per-float and only at the very lightest of them.
- **`low` itself is not always drawable.** Where `low` sits above the floor it is drawable
  and carries exactly one float's share. Where it falls below the floor and off the lattice
  it cannot be returned at all — e.g. a draw over `[-1, 1e30)` never returns exactly `-1`.
- **A vanishingly small side of a hugely lopsided range can get probability exactly zero.**
  A draw over `[-1, 1e30)` returns no negative value at all: the negative side is worth
  about 10⁻³⁰ of the range, which is less than the smallest share the draw can express, and
  it is dropped rather than rounded up to that share. If you need both signs represented, do
  not pair bounds whose magnitudes differ by 30-odd orders of magnitude.

## Negative zero is never returned

A draw never returns `-0f`, whatever you pass. `-0f` can only enter through `low` — a drawn
value is never a negative zero, and `high` is an exclusive bound — and a `low` of `-0f` is
normalised to `+0f` before anything else looks at it. So `RandomUniform(shape, -0f, 0f)`,
`RandomUniform(shape, -0f, -0f)` and an inverted range starting at `-0f` all yield `+0f`
rather than the `-0f` the "returns `low`" rule above would otherwise give. The two values
are numerically equal, so only a bit comparison can see the difference — but the guarantee
is a bit-level one, and it is enforced on the bounds before any drawing happens, so it does
not depend on how an execution provider treats a negative zero.
