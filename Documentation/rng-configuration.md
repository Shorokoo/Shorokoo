# Configuring randomness with `RngConfig`

`RngConfig` decides what every random draw in a model produces: parameter
initialization and runtime feeds (Dropout masks, sampling, in-model noise). It is
**configuration, not architecture**: a model definition never contains a seed, and one
graph can be bound to different configs. (What a uniform draw returns is in
[uniform-draws.md](uniform-draws.md).)

```csharp
var config = new RngConfig { MasterSeed = 42 };

// Inference: bind at concretization.
var concrete = arch.ToConcreteModel(config);

// Training: bind at rig construction — inits AND in-step feeds are keyed.
var rig = TrainingRig.FromScratch(model, loss, optimizer, sampleInputs, hypers, config);
```

## The key tree

One `MasterSeed` derives everything. It splits into two sub-masters: `params`
(initialization, drawn once) and `runtime` (feeds, drawn every execution). Each
stream's key is its sub-master **folded along the consumer's ModelId path**, one
Threefry-2x32 bijection per path index: `child = Bijection(key: parent, counter: index)`.
Keys, split indices and draw positions are full 64-bit values, so distinct indices
always give distinct children.

A consumer **inside a loop** folds one more index per enclosing loop, its trip number,
so a draw executed N times gives N samples; see [Feeds inside loops](#feeds-inside-loops)
and [Init draws inside loops](#init-draws-inside-loops).

Consequences:

- distinct consumers get decorrelated streams;
- a stream's key is reconstructible offline from its ModelId (plus trip numbers inside
  loops), with no draw-order bookkeeping;
- inserting or removing consumers re-keys only the streams whose ModelIds move;
  [`Rng.Pin`](rng-pinning.md) can freeze those;
- changing `MasterSeed` re-randomizes everything coherently.

## The RNG identity is an ordinary parameter: `RngSeed`

A model's runtime RNG identity is one non-trainable parameter, **`RngSeed`**, at the
reserved ModelId `[0]`. Its value encodes the runtime master key, per-stream overrides
and the algorithm id. Each feed's key is **derived in-graph** by a chain of split ops
from `RngSeed` along the feed's ModelId path.

- `ToConcreteArchitecture` creates the value-less `RngSeed` (only if the model has a
  runtime random feed) and wires each feed's chain.
- Binding a config (`ToConcreteModel(config)` / `WithRngConfig`) validates it against
  the model's random surface and writes the `RngSeed` value.

Hence:

**Re-binding is a parameter write.** Changing the seed on a built model changes one
parameter; every draw re-derives, and weights are untouched:

```csharp
var concrete = arch.ToConcreteModel(new RngConfig { MasterSeed = 1 });
// ... later: same weights, different randomness — exact, not approximate.
// (ComputationGraphs are readonly: WithRngConfig returns the re-keyed copy.)
concrete = concrete.WithRngConfig(new RngConfig { MasterSeed = 2 });
```

Re-applying the original config restores the original draws bit-for-bit, on a
**loaded** model too.

Re-binding never re-initializes **weights**. For that, rebuild from the concrete
architecture: `arch.ToConcreteModel(new RngConfig { MasterSeed = … })`. A built model
cannot re-initialize its weights in place.

**Randomness survives save/load.** `RngSeed` is saved as an ordinary initializer and
the chains are part of the graph, so a loaded model draws what it drew before saving,
without a config.

**Training needs nothing special.** The rig binds before loss composition and
autodiff, so forward and recomputed backward copies of a feed derive the same key; a
Dropout's backward mask matches its forward mask.

A model with no runtime random feeds has no `RngSeed`, no chains, and nothing
RNG-related in its saved form.

## What a keyed feed lowers to

At ONNX prep, the key chain and the draw lower to calls of the config's **named RNG
algorithm**: versioned *split* / *uniform* / *normal* functions exported as tagged,
non-inlined ONNX local `FunctionProto`s. `RngSeed` becomes an initializer, and session
constant folding collapses constant chain segments to literal keys, so the chain adds
no per-draw cost (except in a graph with no inputs, which is built without graph
optimizations). Every draw in an exported model is traceable to its function and
the identity initializer, and is bit-identical on every execution provider that
implements integer ops correctly, since key derivation and decoding are pure integer
arithmetic.

## Choosing the generator

`RngConfig.Algorithm` selects the bit generator and the decodes built on it:

- `RngAlgorithm.Threefry2x32` (default): 20 rounds (the Random123 safety-margin
  default).
- `RngAlgorithm.Threefry2x32Rounds13`: 13 rounds (Random123 `threefry2x32x13`), still
  BigCrush-resistant; the lower-margin choice.

They differ only in round count; the decodes ([uniform-draws.md](uniform-draws.md),
[normal-draws.md](normal-draws.md)) and the key tree are identical, so switching never
changes which stream is which, only the numbers drawn. Both collections honor the
choice. A model whose identity names an algorithm this build does not know fails
loudly.

**Why the decodes are integer-only.** ONNX specifies no accuracy for `Log`, `Sqrt`,
`Cos` or `Sin`, so a transcendental normal transform (`√(−2·ln w)·cos(2πu)`) can
differ between conformant providers: in float32 versus binary64, 66% of draws differ,
by up to 4.19e-2 relative. Shorokoo's normal decodes one 64-bit generator value per
element with integer arithmetic alone (a piecewise degree-12 series inverting the
Gaussian CDF), so the value depends only on the bits.

You can switch generators on a built model with `WithRngConfig`, like re-seeding; the
exported model is tagged with the selected algorithm. A loaded model keeps its feed ops,
so a re-bind there may change the algorithm too. Only a graph whose draws were already
**lowered** at ONNX prep has its draw functions baked: a re-bind there may change seeds but
not the algorithm (it fails loudly; rebuild from the architecture).

**The execution counter.** Per-execution variation comes from one model-global counter,
`RngExecutionCounter` (ordinary model state: an int64 scalar starting at 0, +1 per
execution), which concretization wires into every feed. Each execution draws from its
own substream; stream keys are unaffected. Modules never touch it:
`Globals.RandomUniform` is all a consumer writes. Under the training rig the counter
is saved in the checkpoint, so Dropout masks differ per step and a run resumed at step
N draws what the uninterrupted run would. In one-shot inference it is fixed at 0, so
inference is deterministic and stateless. The counter is exact 64-bit state with no
wrap, so a stream never repeats (individual values still collide by chance).

## Feeds inside loops

A feed under a loop gets a ModelId with a `-1` iteration slot, like a parameter created
in a loop: one stream **per iteration**. The runtime iteration index fills that slot as
a split counter in the key chain, so iteration *i*'s stream is the path with `i` in the
slot (e.g. `[loopSlot, i, feedSlot]`): deterministic, resumable and reconstructible
offline, with no per-iteration key storage. Rolled and unrolled loops give the same
keys bit-for-bit.

A feed reached by **calling a module inside a loop** also gets a per-iteration stream,
whether the model was created inside or outside the body and whether it is called as a
model or a module-typed `Function`. Its path is the call site's id extended with one
`(-2, -1)` pair per enclosing loop the callee's own id does not already include (a
model created inside a loop already has that loop's slot). `-2` marks a call site's loop
scope; slots count from 1 (0 is `RngSeed`), so it collides with nothing. Models passed
as a `[Hyper]` model parameter, or taken out of a `ModelSequence`, behave the same way.

A model picked from a sequence at **run time** has its id supplied as split counters:
one `-1` per id component, filled at run time. So a `-1` in a reported path is either
an iteration index (after the loop's slot or a `-2`) or a component of a picked model's
id; the pin skeleton's scope comment names both possibilities. Each model still has its
own stream, but the report cannot enumerate them, `Rng.Pin` cannot name such a model,
and an `Override` on that path must give a concrete value for the `-1`.

Where concretization settles those components (e.g. an unrolled loop), the report
shows the realized stream as `ModelIdPath` and the `-1`-bearing site as `SitePath`,
which is what an `Override` matches against. A slot still open at run time keeps its
`-1` and its row stands for the whole set. A sequence assembled inside a loop therefore
lists one stream per realized trip.

The iteration slot separates the **iterations** of one call site; the call-site
component below separates **call sites**.

The callee's **parameters** do not get the iteration slot: one call site reading one
parameter every iteration is weight sharing. Create the model inside the body to get a
parameter per iteration.

## Calling one model twice

Calling one model *object* at two sites shares its id and parameters, but each draw
must be a fresh sample. So a feed of a model called more than once gets a **call-site
component** at the end of its path: `-3` followed by the call's ordinal (from 0). Like
`-2`, it collides with no slot or `-1`.

Only feeds take it; parameter ids stay shared. A model called once has no call-site
component, so its `Rng.Pin` entries stay valid; adding a second call renames both
calls' streams.

Two call sites of a module-typed `Function` need none of this: each has its own id.

## Init draws inside loops

A `[TrainableParamInitializer]` body may loop (`LoopAPI.Iterate`). Each trip folds its
trip number onto the parameter's init key, so trips are independent samples (see
[nn-library.md](nn-library.md#initializers-shorokoomodulesinitializers)); nested loops
contribute one index each. These per-trip keys are reproducible but have no individual
override addresses; see [Per-stream overrides](#per-stream-overrides).

## Per-stream overrides

`config.Override(RngCollection.Params, [1, 1], seed)` returns a **copy** of the config
with one stream, addressed by its ModelId path from the stream report, pinned to an
explicit seed that replaces its folded key. Configs are immutable (`RngConfig.Default`
included), so chain calls to stack pins. An override survives a later `MasterSeed`
change. Matching is exact and per collection.

```csharp
var config = new RngConfig { MasterSeed = 42 }
    .Override(RngCollection.Params, [1, 1], 1234)
    .Override(RngCollection.Runtime, [2, 0, 1], 5678);
```

Every stream is addressable, including one iteration of a loop feed (e.g. `[1, 2, 1]` =
iteration 2 of the feed at loop slot 1), which re-seeds only that iteration. The
exception is an [init draw inside a loop](#init-draws-inside-loops): a `Params`
override on the parameter re-seeds all its trips together. An override addressing no
stream throws: a `Runtime` one at bind (`WithRngConfig`), a `Params` one at parameter
initialization.

Overrides match the **site pattern**: static components exactly, each `-1` by the
value to select. A value the graph never realizes still matches and simply never
fires; only an address matching no pattern throws. Take `-1` values from the stream
report or the architecture's model ids.

On a built model, changing an override's **value** or the master seed is a parameter
write; changing the override **set** re-wires the chains. Both work on a **loaded**
model (so does changing the algorithm). A graph whose draws were already **lowered** at
ONNX prep accepts new seed *values* only; rebuild from the concrete architecture to
change the set or algorithm.

## Reading a model's identity back: `TryGetRngIdentity`

Ask a built or loaded model what it is bound to:

```csharp
if (model.TryGetRngIdentity() is { } identity)
{
    Console.WriteLine(identity.Algorithm);        // e.g. Threefry2x32
    Console.WriteLine(identity.RunMasterKey);     // the derived runtime master key
    foreach (var o in identity.Overrides)         // path + key, one per overridden stream
        Console.WriteLine(o);
}
```

`null` means no runtime random surface.

An **identity is not a config**:

- **It records a derived key, not `MasterSeed`.** `RunMasterKey` is either the
  `RunMasterSeed` you set or `Fold(MasterSeed, "runtime")` (an XOR with a fixed
  constant), and nothing records which. To reproduce the runtime streams, use
  `new RngConfig { RunMasterSeed = identity.RunMasterKey }`.
- **The `params` collection is not recorded**, since initialization is baked into the
  weights. `Overrides` lists `Runtime` overrides only; re-initializing takes an explicit
  config against the concrete architecture.

## Re-keying one stream: `WithRngOverride`

To change one runtime stream without the original config, override it on the model:

```csharp
// Same weights, same algorithm, same master key, every other stream untouched.
model = model.WithRngOverride(RngCollection.Runtime, [2, 0, 1], 5678);
```

It applies on top of the bound identity, whereas `WithRngConfig` with a fresh config
would re-key every stream. `RngCollection.Params` is rejected, and an address matching
no runtime site pattern throws. It works on loaded models; on a graph whose draws are
already lowered it can re-key an already-overridden stream but not add a new override.

The raw `RngSeed` value (`uint64[]`) is not public API; read it through `RngIdentity`.

## The stream report

`arch.GetRngStreamReport(config)` lists every stream of a concrete architecture: each
parameter's init stream (ModelId path, name, shape, resolved key) and each runtime feed
stream (path, plus its `SitePath` when different). A path containing `-1` stands for an
unenumerable set and has no resolved key; others carry their exact key. The report can
also emit the sparse `Rng.Pin` skeleton (see [Pinning RNG streams](rng-pinning.md)),
grouped by site.

Resolving keys **executes** the in-graph derivations, so a report with a config needs a
working execution provider and is expensive; without a config it resolves nothing.
Pass your own context with `GetRngStreamReport(config, computeContext)`.

## Without a config

A model concretized without a config uses `RngConfig.Default` (master seed 0) for both
collections, exactly as if passed explicitly; it never falls back to backend random
ops. Repeated one-shot inference of such a model repeats its draws; for per-run
variation use `RngConfig.NonDeterministic()`.

`Globals.RandomUniform` / `RandomNormal` / `RandomBits<T>` take no seed; all seeding goes
through `RngConfig`.

`Globals.RandomBits<T>(Vector<int64> shape)` draws raw uniformly random unsigned
integers; `T` must be `uint8`, `uint16`, `uint32` or `uint64`. A `RandomUniform` /
`RandomNormal` site with no ModelId at all (built from op outputs outside any module)
lowers to ONNX `RandomUniformLike` / `RandomNormalLike`, but a `RandomBits` site there
fails at ONNX prep. Draw raw bits inside a concrete, id-bearing model.

A site that has a ModelId but no wired derivation chain (e.g. a `ConcreteArchitecture`
executed before `ToConcreteModel`) is an error for every feed kind:

> `FastLowerRandomOps: the shrk_RandomUniform feed at ModelId [...] is id-bearing but has no key
> derivation chain ...`

Call `ToConcreteModel()` on the architecture and execute that. (A `[Module]`'s own
output is refused earlier as un-lowered; see
[inference.md](inference.md#running-a-module).)

## Choosing seeds

- `RngConfig.Default`: master seed 0; deterministic, and what "no config" means.
- `new RngConfig { MasterSeed = s }`: deterministic under your seed.
- `RngConfig.NonDeterministic()`: a fresh master from system entropy; the chosen seed
  is stored on the object so it can be recorded.

To give two parameters identical initial values (e.g. to match a reference layer),
override both streams to the same seed with `Override(RngCollection.Params, path, s)`.
