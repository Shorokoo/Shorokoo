# What a normal draw returns

Related: [rng-configuration.md](rng-configuration.md) · [uniform-draws.md](uniform-draws.md) · [nn-library.md](nn-library.md) · [core-types.md](core-types.md) · [limitations.md](limitations.md)

Every normally distributed value in Shorokoo comes out of one draw: `Globals.RandomNormal`,
the `Normal` / `NormalDist` / `XavierNormal` / `KaimingNormal` / `XavierNormalGain` /
`KaimingNormalGain` / `LeCunNormal` / `TruncatedNormal` initializers, and the Gaussian that
`Orthogonal` iterates from. This page is that draw's contract: which values it can return,
how finely it resolves the magnitude axis, where it stops, and its imperfections. For
*which* values a given draw produces (seeds, streams, reproducibility) see
[Configuring randomness](rng-configuration.md).

Everything below holds under both `RngAlgorithm` values, which differ only in the bit
generator's round count (see
[Choosing the generator](rng-configuration.md#choosing-the-generator)).

```csharp
var noise  = RandomNormal(Vector(4L, 8L));                // float32, standard normal
var scaled = RandomNormal(Vector(4L, 8L), 1f, 0.02f);     // float32, N(1, 0.02)
var init   = NormalDist.Init([Scalar(4L), Scalar(8L)], Scalar(0f), Scalar(0.02f));  // in-graph
```

## Facts

- The result dtype is always **`float32`**, whatever `mean` and `scale` are.
- **One 64-bit generator value per element.** The top bit is the sign, the low 63 a position
  on the *magnitude axis*. There is no rejection and no pairing, so no two elements share a
  generator value and each element's value depends only on its own position in the stream.
- The draw is **exactly symmetric**. The magnitude is sampled and the sign applied
  afterwards, so `+a` and `-a` are equally likely and own mirror-image intervals of the axis.
  The set of values a draw can return is closed under negation.
- It **rounds to nearest**. A draw lands on the `float32` nearest the real value its position
  names, with mean distance a quarter of a relative ulp (the best any sampler returning a
  `float32` can do) and no pull towards zero. The uniform draw, by contrast, rounds down (see
  [the range is addressed, not scaled](uniform-draws.md#the-range-is-addressed-not-scaled)).
- The magnitude is **bounded**. No draw exceeds **8**, and magnitudes below 2⁻³⁹ ride an
  even lattice rather than the float grid. See
  [how finely the magnitude axis is resolved](#how-finely-the-magnitude-axis-is-resolved) and
  [where the axis stops](#where-the-axis-stops-no-draw-exceeds-8).
- The whole decode is integer arithmetic, so the same key yields identical values bit for
  bit on every execution provider and in an exported ONNX model; see
  [the same bits everywhere](#the-same-bits-everywhere).
- `mean` and `scale` may be compile-time literals or graph scalars computed in-graph: the
  two `RandomNormal` overloads in
  [core-types.md](core-types.md#factory-helpers-using-static-shorokooglobals). Both forms use
  the same draw and carry the same guarantees. Graph-scalar parameters cannot be expressed as
  ONNX attributes, so they also need a concrete model built through
  [`ToConcreteModel`](rng-configuration.md), not a bare architecture. The
  [initializers](nn-library.md#initializers-shorokoomodulesinitializers) that take `mean` and
  `scale` as `Init` arguments do not go through that overload: they draw standard-normal and
  apply the shift and scale as ordinary graph arithmetic, so they need no concrete model.

## The magnitude is addressed, the sign applied after

The construction: **the low 63 bits address a magnitude by inverting the half-normal CDF,
and the top bit gives that magnitude a sign.** Each reachable `float32` magnitude owns the
run of positions whose real values round to it (cells run away from zero, with boundaries
at the midpoints between neighbouring floats), so a magnitude comes out with the normal mass
of the interval it stands for, and the finest share the axis can express is one position in
2⁶³. Both signs address the same axis, so `+a` and `-a` are one cell mirrored and the
symmetry is exact.

Most standard libraries instead transform uniform values through `√(−2·ln w)·cos(2πu)`.
Addressing magnitudes directly gives two guarantees:

- **The value is fixed by the bits alone.** No `Log`, `Sqrt`, `Cos` or `Sin` appears in the
  decode, so no execution provider's transcendental accuracy can move a result (see
  [below](#the-same-bits-everywhere)).
- **Rounding is to nearest, and the symmetry is exact.** Both are properties of how the axis
  is cut into cells, not of a formula's numerics.

## Mean and scale are applied afterwards

`RandomNormal(shape, mean, scale)` — and `NormalDist`, and every initializer with a
prescribed standard deviation — computes `z·scale + mean` from a standard draw `z` in
ordinary `float32` arithmetic. Everything else on this page describes `z`; the affine step
rounds as a `float32` multiply and add round. Two consequences:

- The 8-sigma bound scales with it: a draw never leaves `mean ± 8·scale`.
- With `mean = 0` the exact symmetry survives the scaling, since negating a `float32` product
  is exact. A non-zero `mean` is an ordinary addition and rounds as one.

Nothing is re-drawn or rejected, so `TruncatedNormal`'s `[−2, 2]` is a clamp rather than
rejection sampling
([nn-library.md](nn-library.md#initializers-shorokoomodulesinitializers)).

## How finely the magnitude axis is resolved

A **weight class** is one power of two of magnitude: 2²³ `float32` values that share one
ulp, the same classes the uniform draw counts, defined in
[how finely the range is resolved](uniform-draws.md#how-finely-the-range-is-resolved). The
normal draw resolves **42 successive classes**, from 2⁻³⁹ up to 8. Call the bottom of that
span the *floor* and the top the *cap*.

| Region | Magnitudes | What a draw returns there | Mass |
|---|---|---|---|
| lattice | below the floor, 2⁻³⁹ (1.818989e-12) | points of an even lattice whose step is the floor class's spacing — represented, but not individually addressable | 1.4513e-12, about 1 draw in 690 billion |
| resolved | 2⁻³⁹ up to 8, the 42 weight classes between | individual `float32` values, each with the mass of the interval it stands for — except above 7.6008, where a float's cell is worth under one position; 577,209 of them get none, the lowest of them 7.6011825 | all the rest |
| cap | 8 and beyond | exactly `8.0f` | 1.244e-15, about 1 draw in 800 trillion |

Counting values: **720,265,872** of the 4,278,190,080 finite `float32` values (**16.8%**)
are individually reachable, a set closed under negation: 351,744,327 resolved magnitudes and
8,388,607 lattice points on each sign, plus ±`8.0f` and both zeros.

**Below the floor**, truncation spends resolution, not fairness. Across that region the
normal density is constant to within 2⁻⁷⁸, so an evenly spaced lattice *is* the correct
distribution, and the region carries exactly its due mass. A draw there returns a lattice
point instead of the neighbouring float a full-resolution draw would have named; for any use
that treats these values as numbers rather than bit patterns the two are interchangeable.

## Where the axis stops: no draw exceeds 8

The resolved window ends at the start of weight class 130, which is the value 8. Every
position at or past 8 sigma decodes to exactly `8.0f`, so that single float carries the whole
tail beyond it: **1.244e-15** of the mass, about **one draw in 800 trillion**.

This is a hard clip on the tail, at a magnitude you will not reach by accident. A standard
draw never returns an infinity, a NaN, or any value outside `[−8, 8]`; a scaled draw never
leaves `mean ± 8·scale`, so e.g. `KaimingNormal` on a fan-in of 1024 draws
`N(0, √(2/1024))` and cannot produce a weight past 8·√(2/1024) ≈ 0.354. Initializing a
billion parameters from normal draws gives about one chance in a million that any single
element lands on the cap.

It is observable in code that looks at the far tail: importance sampling weighted into it,
extreme-value estimation, a test that asks how often `|z| > t` for large `t`. Past 8 there
is a point mass on `8.0f` instead of a continuing tail, and every sigma level beyond 8
reports the same count.

## The same bits everywhere

The whole decode, from generator bits to the `float32` returned, is integer arithmetic. Any
execution provider that implements the integer operations correctly returns the same values
from the same key, bit for bit, and an exported ONNX model draws what Shorokoo drew. The same
seed yields identical values under ONNX Runtime's CPU provider, under the
[Quick Execution Engine](limitations.md#quick-execution-engine-value-computation-is-bounded),
and in an exported model.

The transcendental construction, `√(−2·ln w)·cos(2πu)`, cannot guarantee this: ONNX
specifies no accuracy for `Log`, `Sqrt`, `Cos` or `Sin`, so two conformant providers can
return different normals from the same key. Evaluating that formula in float32 and in
binary64 makes **66%** of its draws differ, the worst by **4.19e-2** relative.
