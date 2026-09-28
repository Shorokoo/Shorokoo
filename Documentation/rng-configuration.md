# Configuring randomness with `RngConfig`

`RngConfig` decides what every random draw in a model produces: parameter initialization
and runtime feeds (Dropout masks, sampling, in-model noise) alike. It is **configuration,
not architecture**: a model definition never contains a seed, and the same graph can be
bound to different configs at different times. (This page covers *which* values a draw
produces; what a uniform draw returns, including the interval, edge cases and resolution,
is in [uniform-draws.md](uniform-draws.md).)

```csharp
var config = new RngConfig { MasterSeed = 42 };

// Inference: bind at concretization.
var concrete = arch.ToConcreteModel(config);

// Training: bind at rig construction — inits AND in-step feeds are keyed.
var rig = TrainingRig.FromScratch(model, loss, optimizer, sampleInputs, hypers, config);
```

## The key tree

One `MasterSeed` derives everything. Two sub-masters split off it, one for the `params`
collection (initialization noise, drawn once at materialization) and one for the `runtime`
collection (feeds, drawn every execution). Each stream's key is the sub-master **folded
along the consumer's ModelId path**: one Threefry-2x32 bijection per path index,
`child = Bijection(key: parent, counter: index)`. A key, a split index and a draw position
are each a full 64-bit value with no narrowing, so distinct indices give distinct children
over the entire range.

A consumer **inside a loop** folds one more index per enclosing loop: its trip number. For a
runtime feed that index is part of the ModelId path (the `-1` slots below); for a parameter
initializer's draw it is folded onto the parameter's key by the same bijection. Either way a
draw executed N times gives N samples, not one repeated; see
[Feeds inside loops](#feeds-inside-loops) and [Init draws inside loops](#init-draws-inside-loops).

Because the key tree *is* the ModelId tree:

- distinct consumers get decorrelated streams;
- a stream's key is reconstructible offline from its ModelId alone, plus the trip number
  for a consumer inside a loop (no draw-order bookkeeping);
- inserting or removing consumers re-keys only the streams whose ModelIds move, and
  [`Rng.Pin`](rng-pinning.md) can freeze those against refactoring;
- changing `MasterSeed` re-randomizes everything at once, coherently.

## The RNG identity is an ordinary parameter: `RngSeed`

A model's runtime RNG identity lives in one ordinary non-trainable parameter,
**`RngSeed`**, at the reserved ModelId `[0]` (slot 0 is never assigned to anything else).
Its value encodes the runtime master key, any per-stream overrides, and the algorithm id.
Every feed's key is **derived in-graph**: a chain of split operations rooted at `RngSeed`,
one split per element of the feed's ModelId path (the fold above, as graph ops).

- `ToConcreteArchitecture` creates the value-less `RngSeed` parameter (only if the model has
  at least one runtime random feed) and wires each feed's derivation chain, as it realizes
  the model's parameters.
- Binding a config (`ToConcreteModel(config)` / `WithRngConfig`) validates it against the
  model's random surface and **is the `RngSeed` parameter's initialization**: it writes the
  identity value, as `ToConcreteModel` fills weights.

Three properties follow:

**Re-binding is a parameter write.** You can change the master seed *after*
`ToConcreteModel`, on the built model: one parameter value changes, every draw re-derives
from it, and trained or loaded weights are untouched:

```csharp
var concrete = arch.ToConcreteModel(new RngConfig { MasterSeed = 1 });
// ... later: same weights, different randomness — exact, not approximate.
// (ComputationGraphs are readonly: WithRngConfig returns the re-keyed copy.)
concrete = concrete.WithRngConfig(new RngConfig { MasterSeed = 2 });
```

Re-applying the original config restores the original draws bit-for-bit. The saved model
keeps the symbolic derivation chains, so this holds on a **loaded** model too: load, call
`WithRngConfig`, and every draw is re-keyed.

Re-binding never touches **weights**. To re-initialize the trainable parameters under a
different seed, keep the concrete architecture and rebuild from it with
`arch.ToConcreteModel(new RngConfig { MasterSeed = … })`, which re-runs every parameter's
keyed initializer. A built model cannot re-initialize its weights in place: its initializer
functions are gone once values are bound.

**Randomness survives save/load.** `RngSeed` serializes as an ordinary initializer (no
reserved names, no side channels) and the chains are part of the graph, so a loaded model
draws what it drew before saving, with no config object needed on the loading side.

**Training needs nothing special.** The rig binds the concrete architecture at the same
point inference does, *before* loss composition and autodiff. `RngSeed` and the chains
carry into the training-step graph, and the forward and any recomputed backward copy of a
feed derive the same key, so a Dropout's backward mask matches its forward mask.

A model with no runtime random feeds carries none of this: no `RngSeed`, no chains,
nothing RNG-related in its saved form.

## What a keyed feed lowers to

At ONNX prep a feed's key chain feeds the draw call directly: the chain's splits and the
draw both lower to calls of the config's **named RNG algorithm**, a versioned set of
functions (kinds *split* / *uniform* / *normal*) that export as tagged, non-inlined ONNX
local `FunctionProto`s. `RngSeed` becomes an ordinary initializer, and in a compiled model
the runtime's session-build constant folding collapses the constant chain segments to
literal keys, so the chain adds no per-draw cost. (A graph with no inputs at all is built
without graph optimizations, so nothing is folded there; it is a constant computed once and
discarded, which is cheaper to run than to fold.) The exported model's randomness is
therefore identifiable (you can point at the function in the ONNX file that produced any
draw, and at the initializer carrying the identity it derived from), deterministic, and
portable across execution providers: every step from key to drawn bit pattern is integer
arithmetic, so any provider that implements the integer ops correctly returns the same
bits.

## Choosing the generator

`RngConfig.Algorithm` selects the bit generator **and the decodes that turn its bits into
values**, since both decide what a draw returns:

- `RngAlgorithm.Threefry2x32` (default): Threefry-2x32, 20 rounds (the Random123
  safety-margin default).
- `RngAlgorithm.Threefry2x32Rounds13`: the reduced 13-round bit generator (Random123
  `threefry2x32x13`), still BigCrush-resistant, 7 rounds fewer; the lower-margin choice.

The two differ only in the bit generator's round count. The decodes are identical (the
uniform's in [uniform-draws.md](uniform-draws.md), the normal's in
[normal-draws.md](normal-draws.md)), and so is the key tree. The recorded identity names
the algorithm a model is bound to, and a model whose identity names an algorithm this build
does not know fails loudly instead of falling back to another.

**Why the decodes are integer-only.** A normal is commonly built by transforming uniform
values through `√(−2·ln w)·cos(2πu)`, four float32 transcendentals deep. ONNX specifies no
accuracy for `Log`, `Sqrt`, `Cos` or `Sin`, so two conformant execution providers can return
different normals from the same key; evaluating that formula in float32 and in binary64
makes 66% of draws differ, the worst by 4.19e-2 relative. Shorokoo's normal spends one
64-bit generator value per element and decodes it by **integer arithmetic alone**: a table
of pieces, each inverting the Gaussian CDF over its run of magnitudes with a degree-12
series, with no transcendental anywhere. The value is fixed by the bits alone, on every
execution provider and in an exported model.

Both algorithms **share one key tree**: a stream's key is derived the same way regardless of
algorithm, so switching `Algorithm` never changes which stream is which; the same stream
draws different numbers. Parameter initialization and runtime feeds both honor the choice.
Because binding is a parameter write, you can switch generators on a built (in-memory) model
the same way you re-seed it (`concrete = concrete.WithRngConfig(...)`), and the exported
model calls, and is tagged with, the selected algorithm's functions. A **loaded** model's
draw functions are already baked, so a re-bind on it may change seed values but not the
algorithm (that re-bind fails loudly; rebuild from the architecture instead).

Per-execution variation comes from a separate **execution counter**, not from the
configured key, and the RNG system manages it. Concretization injects one model-global
execution counter (`RngExecutionCounter`, ordinary model state, an int64 scalar initialized
to 0 and advanced by 1 per execution) and wires it into every feed as the draw's substream
index. A draw folds the whole 64-bit counter into the stream key instead of spending a
counter word on it, so execution *n* draws from its own substream of that stream and the
element index gets the whole counter. (The fold is a draw-internal step, not a node of the
configurable key tree: a stream's key is unchanged by it, and unlike a key split it runs at
the selected algorithm's own round count.) Modules never touch it; `Globals.RandomUniform`
is all a consumer writes. Under the training rig the counter is saved in the checkpoint, so
Dropout masks differ per step and a run resumed at step N draws what the uninterrupted run
would; in one-shot inference it is baked at 0, so inference is deterministic and stateless.
One counter serves all feeds because sites are already decorrelated by their stream keys;
it costs the checkpoint a single scalar. The counter is 64-bit state end to end (it
increments exactly, and the whole value reaches the draw), so there is no saturation or
wrap point past which a mask would repeat an earlier one. (Individual mask *values* still
collide as often as chance dictates, since `float32` holds finitely many values; the stream
does not repeat.)

## Feeds inside loops

A feed under a loop takes a ModelId with a `-1` iteration slot (like a parameter created in
a loop): one stream **per iteration**. The **runtime iteration index enters the feed's key
derivation chain as a split counter** at that position, so iteration *i*'s stream is the one
at the realized path with `i` in the iteration slot (e.g. `[loopSlot, i, feedSlot]`), its
key the runtime master folded along that full path: deterministic, resumable, and
reconstructible offline from the path alone. No per-iteration key storage exists (the
derivation is one extra bijection per iteration, negligible next to the draw), and the
streams need no enumeration up front. This works identically whether the loop survives to
runtime or is unrolled at concretization (each copy resolves to the same key, bit-for-bit).

A feed reached by **calling a module from inside a loop** takes that slot too, whether the
model was created inside the body or outside it, and by either call form (a model call or a
module-typed `Function`): the draw executes once per iteration, and each execution is a new
sample. The call site's position supplies the scope, so the feed lands under the call site's
id extended with one `(-2, -1)` pair per enclosing loop the callee's identity does not
already account for; a model created inside a loop already carries that loop's slot and
takes no second one. The `-2` marks a call site's loop scope: it is neither a slot (slots
count from 1, with 0 reserved for `RngSeed`) nor the `-1` an iteration fills, so it cannot
collide with anything the callee owns, and as a constant it does not move when the callee
gains or loses a consumer. A model passed as a `[Hyper]` model parameter takes the same
route, since the model bound to the parameter is what gets reparented, and so does one taken
out of a `ModelSequence`.

A model the sequence position picks at **run time** cannot have its own id written into the
path, because which model it is is unknown until the graph runs. Its id arrives as split
counters: the path carries one `-1` per component of it, filled from the model at run time
as a loop's `-1` is filled from the iteration index. So a `-1` in a reported path always
means "a component resolved while the graph runs": an iteration index (following the loop's
own slot, or the `-2` where the scope came from a call site) or one component of a picked
model's id. The path does not say which, so the pin skeleton's scope comment names both. The
stream is still per-model; the *report* cannot enumerate the models, since it lists what the
graph says statically. Consequently `Rng.Pin` cannot name such a model, and an `Override`
addressing that path must give a concrete value for the `-1` (the reported path with the
`-1` in it is not itself an address).

Where concretization *does* settle the components (an unrolled loop leaves its indices as
constants, so a model created on one trip and reached outside it is one the graph can
name), the report says which stream each feed is, not which site it sits at. Such a feed
carries both: its `ModelIdPath` is the realized stream, and its `SitePath` the `-1`-bearing
site the key chain was built from and an `Override` matches against. A slot still open when
the graph runs keeps its `-1`, and its row stands for the whole unbounded set, as above. A
sequence assembled *inside* a loop takes this route: its models are created per trip, so
their feeds each name a realized trip instead of collapsing onto one stream that names no
model.

Reaching a model out of a `ModelSequence` therefore keys its feeds as calling that model
directly does.

The slot separates the **iterations** of one call site. Separating **call sites** is the job
of the call-site component described below, inside a loop and outside alike.

The callee's **parameters** do not take the iteration slot. One call site reading one
parameter on every iteration is weight sharing, which a model created outside the loop
already gives; create the model inside the body to get a parameter per iteration.

## Calling one model twice

Calling the same model *object* at two sites shares its id, and so its parameters; that is
the purpose of calling it twice. A draw is not a value it holds, though: each execution of a
draw is a fresh sample, so the two sites must not share a stream. A feed of a model called
more than once therefore carries a **call-site component** at the end of its path: `-3`,
followed by the call's ordinal among that model's calls, counting from 0. Like `-2`, `-3` is
negative and cannot collide with a slot (slots count from 1, with 0 reserved for `RngSeed`)
or with an iteration's `-1`.

Only the feeds take it; the model's **parameter** ids stay collapsed, since sharing weights
across the calls is the point. Only a model actually called more than once takes it, so a
single-call model's path has no call-site component and a `Rng.Pin` recorded against it
stays valid. Adding a second call splits the stream and renames both halves, because one
stream cannot become two while either keeps the old name.

Two call sites of a module-typed `Function` need none of this: each mints its own id.

## Init draws inside loops

A `[TrainableParamInitializer]` body may loop (`LoopAPI.Iterate`), and a draw inside that
body executes once per trip. Each trip folds its trip number onto the parameter's init key,
so the trips are independent samples: the same rule a feed in a loop follows above, and the
one [nn-library.md](nn-library.md#initializers-shorokoomodulesinitializers) states for
initializer authors. Every enclosing loop contributes an index, so a draw nested two deep
varies with both.

The per-trip keys hang off the parameter's own stream, not off ModelId slots of their own.
They are reproducible for a config like everything else, but have no individual override
addresses; see [Per-stream overrides](#per-stream-overrides).

## Per-stream overrides

`config.Override(RngCollection.Params, [1, 1], seed)` returns a **copy** of the config with
one stream pinned to an explicit seed, replacing the fully folded key for that stream only.
The stream is addressed by its consumer's ModelId path, as listed by the stream report.
Configs are immutable: the receiver (including `RngConfig.Default`) never changes, so chain
`Override` calls to stack pins and keep the result. Because the override replaces the
*result* of the fold, it survives a later `MasterSeed` change. Matching is exact and
per-collection.

```csharp
var config = new RngConfig { MasterSeed = 42 }
    .Override(RngCollection.Params, [1, 1], 1234)
    .Override(RngCollection.Runtime, [2, 0, 1], 5678);
```

Every stream is a valid override address, including the per-iteration streams of a loop
feed (e.g. `[1, 2, 1]` = iteration 2 of the feed at loop slot 1); overriding one iteration
re-seeds only that iteration, and sibling iterations keep their derived keys. An
[init draw inside a loop](#init-draws-inside-loops) is the exception: its per-trip keys hang
off the parameter's key, not off ModelId slots, so no address names one trip. A `Params`
override on that parameter re-seeds all its trips together, coherently. An override that
addresses no stream throws: a `Runtime` override at bind (`WithRngConfig`), a `Params`
override at parameter initialization.

An override is matched against the **site pattern**, not a realized stream: static
components exactly, and each `-1` by the concrete value to select. It then routes
structurally in the feed's derivation chain, so it works for any value that position takes:
any iteration the loop executes, any model the sequence position picks. A value the graph
never realizes still matches the pattern, so it is accepted and never fires; only an address
matching no *pattern* throws. Give `-1` positions values read off the stream report or the
model ids in the architecture, not guesses.

Changing an override's **value** (or the master seed) on a built model is a pure parameter
write; changing the override **set** re-wires the derivation chains. A saved model keeps its
feed ops, so a re-bind on a **loaded** model may change the set (and the algorithm) too. A
graph whose random draws have already been **lowered** to baked draw-function calls at ONNX
prep has fixed routing and generator: a re-bind may still change seed *values* but not the
override set or the algorithm. Rebuild from the concrete architecture for that.

## Reading a model's identity back: `TryGetRngIdentity`

A model carries its runtime RNG identity, so you can ask a built or loaded model what it is
bound to without the config that produced it:

```csharp
if (model.TryGetRngIdentity() is { } identity)
{
    Console.WriteLine(identity.Algorithm);        // e.g. Threefry2x32
    Console.WriteLine(identity.RunMasterKey);     // the derived runtime master key
    foreach (var o in identity.Overrides)         // path + key, one per overridden stream
        Console.WriteLine(o);
}
```

`null` means the model has no runtime random surface (no `RngSeed`, nothing to report).

An **identity is not a config**:

- **It records a derived key, not `MasterSeed`.** `RunMasterKey` is the runtime sub-master
  the feeds fold from: the `RunMasterSeed` you set, or else `Fold(MasterSeed, "runtime")`, an
  XOR against a fixed constant from which a master seed can be read back. Nothing records
  which of the two a model has, so the recorded key is not in general the seed a caller
  typed. To reproduce a model's runtime streams under a fresh config, pass the recorded key
  as `RunMasterSeed` (`new RngConfig { RunMasterSeed = identity.RunMasterKey }`) instead of
  working back to a master seed.
- **The `params` collection is not recorded.** Initialization randomness is drawn once and
  baked into the weights, so nothing in a saved model consumes it. `Overrides` lists
  `Runtime` overrides only, and re-running initialization under a chosen seed takes an
  explicit config against the concrete architecture.

## Re-keying one stream: `WithRngOverride`

To change one runtime stream on a model whose config you no longer have, override it on the
model directly, the graph-level counterpart of `RngConfig.Override`:

```csharp
// Same weights, same algorithm, same master key, every other stream untouched.
model = model.WithRngOverride(RngCollection.Runtime, [2, 0, 1], 5678);
```

The override is applied on top of the model's *bound identity*, so nothing else moves;
building an `RngConfig` from scratch and calling `WithRngConfig` would replace the whole
identity and silently re-key every other stream. `RngCollection.Params` is rejected (there is
no recorded init tier to override), and an address matching no runtime site pattern throws,
as through `RngConfig.Override`. It works on a loaded model too (the saved feed ops let the
override set change), with the exception noted above: a graph whose draws are already
lowered can re-key a stream it *already* overrides, but cannot gain a new override.

The raw encoded identity (the `RngSeed` parameter's `uint64[]` value) is not public API. Its
layout is an implementation detail, and a file does not record which layout it was written
with, so read the identity through `RngIdentity`.

## The stream report

`arch.GetRngStreamReport(config)` inventories every stream of a concrete architecture: the
init stream of each parameter (ModelId path, name, shape, resolved key) and one row per
runtime feed stream (its path, plus the `-1`-bearing `SitePath` it derives from when the two
differ). A path with a `-1` in it stands for a set the graph cannot enumerate and carries no
resolved key; one the graph can name carries its exact key, whether static all along or
settled by an unrolled loop. The report can also emit the sparse `Rng.Pin` skeleton for
freezing streams before a refactor (see [Pinning RNG streams](rng-pinning.md)); the skeleton
groups by site, so realized streams of one site stay one entry.

Resolving the keys **executes** each stream's in-graph derivation (Shorokoo computes no
randomness on the host), so a report *with* a config is an expensive call that needs a
working execution provider. A report without one resolves nothing. Pass your own context if
you have one: `GetRngStreamReport(config, computeContext)`.

## Without a config

There is no unkeyed concrete model. A model concretized without a config gets the **default
deterministic identity**, `RngConfig.Default` (master seed 0), for both collections:
parameters initialize under it and feeds draw keyed Threefry under it, as if you had passed
it explicitly. "No config" means seed 0; it never falls back to backend random ops. Repeated
one-shot inference of a no-config model therefore repeats its draws bit-for-bit; for per-run
variation, use `RngConfig.NonDeterministic()`.

There is no per-site seed either: `Globals.RandomUniform` / `RandomNormal` /
`RandomBits<T>` take no seed parameter. All seeding goes through `RngConfig`, addressed by
ModelId when a single stream needs pinning.

`Globals.RandomBits<T>(Vector<int64> shape)` is the third keyed feed: raw uniformly random
unsigned integers, with `T` constrained to `uint8` / `uint16` / `uint32` / `uint64`; any
other width is rejected. Unlike the float feeds it has **no unkeyed fallback**. A
`RandomUniform` / `RandomNormal` site that carries no ModelId at all (a draw assembled from
op outputs outside any module; a module body's draws get their ids where the body is built)
lowers to ONNX `RandomUniformLike` / `RandomNormalLike`, but a bits site in that position
fails loudly at ONNX prep, since a bit pattern is meaningful only under a stream key. Draw
raw bits inside a concrete, id-bearing model.

A site that *does* carry a ModelId but whose derivation chain is not yet wired (a draw inside
an un-run initializer body, so a `ConcreteArchitecture` executed before `ToConcreteModel`) is
a hard error for every feed kind, the float ones included:

> `FastLowerRandomOps: the shrk_RandomUniform feed at ModelId [...] is id-bearing but has no key
> derivation chain ...`

The site belongs to the keyed streams, so lowering it to a backend random op would silently
trade the model's reproducibility for unkeyed randomness. Call `ToConcreteModel()` on the
architecture and execute that. (A `[Module]`'s own output never reaches this error: it is
refused earlier as un-lowered; see [inference.md](inference.md#running-a-module).)

## Choosing seeds

- `RngConfig.Default`: master seed 0; fully deterministic, and what "no config" means.
- `new RngConfig { MasterSeed = s }`: deterministic under your seed.
- `RngConfig.NonDeterministic()`: a fresh master from system entropy each run; the chosen
  seed is fixed on the object so the run stays internally consistent and can be recorded.

To give two parameters identical initial values (e.g. to match a hand-built reference
layer), override each stream to the same seed: `Override(RngCollection.Params, path, s)` on
both.
