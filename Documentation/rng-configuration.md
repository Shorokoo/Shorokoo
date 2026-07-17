# Configuring randomness with `RngConfig`

`RngConfig` is the single object that decides what every random draw in a model produces —
parameter initialization and runtime feeds (Dropout masks, sampling, in-model noise) alike.
It is **configuration, not architecture**: a model definition never contains a seed, and the
same graph can be bound to different configs at different times.

```csharp
var config = new RngConfig { MasterSeed = 42 };

// Inference: bind at concretization.
var concrete = arch.ToConcreteModel(config);

// Training: bind at rig construction — inits AND in-step feeds are keyed.
var rig = TrainingRig.FromScratch(model, loss, optimizer, sampleInputs, hypers, config);
```

## The key tree

One `MasterSeed` derives everything. Two sub-masters split off it — one for the `params`
collection (initialization noise, drawn once at materialization) and one for the `runtime`
collection (feeds, drawn every execution) — and each individual stream's key is the
sub-master **folded along the consumer's ModelId path**: one Threefry-2x32 bijection per path
index, `child = Bijection(key: parent, counter: index)`.

Because the key tree *is* the ModelId tree:

- distinct consumers get decorrelated streams for free;
- a stream's key is reconstructible offline from its ModelId alone (no draw-order bookkeeping);
- inserting or removing consumers only re-keys the streams whose ModelIds move — and
  [`Rng.Pin`](rng-pinning.md) can freeze those against refactoring;
- changing `MasterSeed` re-randomizes everything at once, coherently.

## The RNG identity is an ordinary parameter: `RngSeed`

A model's entire runtime RNG identity lives in one ordinary non-trainable parameter,
**`RngSeed`**, at the reserved ModelId `[0]` (slot 0 is never assigned to anything else).
Its value encodes the runtime master key, any per-stream runtime overrides, and
the algorithm id. Every feed's key is then **derived in-graph**: a chain of split
operations rooted at the `RngSeed` parameter, one split per element of the feed's ModelId
path — the fold described above, spelled out as graph ops.

- `ToConcreteArchitecture` creates the value-less `RngSeed` parameter (if — and only if —
  the model has at least one runtime random feed) and wires each feed's derivation chain,
  exactly as it realizes the model's parameters.
- Binding a config (`ToConcreteModel(config)` / `ApplyRngConfig`) validates it against the
  model's random surface and **is the `RngSeed` parameter's initialization**: it writes the
  identity value, the same way `ToConcreteModel` fills weights.

Three properties follow directly:

**Re-binding is a parameter write.** You can change the master seed *after*
`ToConcreteModel`, on the already-built model — one parameter value changes, every draw
re-derives from it, trained or loaded weights are untouched:

```csharp
var concrete = arch.ToConcreteModel(new RngConfig { MasterSeed = 1 });
// ... later: same model object, different randomness — exact, not approximate.
concrete.ApplyRngConfig(new RngConfig { MasterSeed = 2 });
```

Re-applying the original config restores the original draws bit-for-bit — and because the
saved model keeps the symbolic derivation chains, this holds equally on a **loaded** model:
load, `ApplyRngConfig`, and every draw is re-keyed by construction.

**Randomness survives save/load.** `RngSeed` serializes as an ordinary initializer (no
reserved names, no side channels) and the chains ride the graph, so a loaded model draws
exactly what it drew before saving — no config object needed on the loading side.

**Training needs nothing special.** The rig binds the concrete architecture at the same
shared point inference uses, *before* loss composition and autodiff. The `RngSeed`
parameter and the chains ride along into the training-step graph, and the forward and any
recomputed backward copy of a feed derive the same key — so e.g. a Dropout's backward
mask matches its forward mask by construction.

A model with no runtime random feeds carries none of this: no `RngSeed`, no chains,
nothing RNG-related in its saved form.

## What a keyed feed lowers to

At ONNX prep a feed's wired key chain feeds the draw call directly: the chain's splits and
the draw both lower to calls of the config's **named RNG algorithm** — a versioned set of
functions (kinds *split* / *uniform* / *normal*) that export as tagged, non-inlined ONNX
local `FunctionProto`s. `RngSeed` becomes an ordinary initializer, and the runtime's
session-build constant folding collapses the constant chain segments to literal keys — so
the per-draw cost is what it always was. The exported model's randomness is therefore
deterministic, portable across execution providers, and identifiable: you can point at the
function in the ONNX file that produced any draw — and at the initializer that carries the
identity it derived from.

## Choosing the generator

`RngConfig.Algorithm` selects the bit generator:

- `RngAlgorithm.Threefry2x32` (default) — Threefry-2x32, 20 rounds; the Random123
  safety-margin default.
- `RngAlgorithm.Threefry2x32Rounds13` — the reduced 13-round variant (Random123
  `threefry2x32x13`): still BigCrush-resistant, ~35% cheaper — the faster, lower-margin choice.

All algorithms **share one key tree**: a stream's key is derived the same way regardless of
generator, so switching `Algorithm` never reshuffles which stream is which — the same stream
simply draws different numbers. Both parameter initialization and runtime feeds honor the
choice. Because binding is a parameter write, you can switch generators on an
already-built (in-memory) model the same way you re-seed it (`concrete.ApplyRngConfig(...)`),
and the exported model calls — and is tagged with — the selected algorithm's functions. A
**loaded** model's draw functions are already baked, so on it a re-bind may change seed
values but not the algorithm (that re-bind fails loudly; rebuild from the architecture
instead).

Per-execution variation is carried by a separate **drawBase** counter, not by the key — and
the RNG system manages it itself: concretization injects one model-global execution counter
(`RngExecutionCounter`, ordinary model state, an int64 scalar initialized 0 and advanced +1
per execution) and wires it into every feed. Modules never touch it — `Globals.RandomUniform`
is all a consumer writes. Under the training rig the counter rides the checkpoint, so Dropout
masks differ per step and a resumed run at step N draws exactly what the uninterrupted run
would; in one-shot inference it is baked at 0, so inference stays deterministic and
stateless. One counter serves all feeds because sites are already decorrelated by their
stream keys; it costs the checkpoint a single scalar. Framework-managed counters like this
one are int64 state end-to-end, so incrementing stays exact at any step count — there is no
float32-style saturation point past which masks would stop varying.

## Feeds inside loops

A feed under a loop takes a ModelId with a `-1` iteration slot (exactly like a parameter
created in a loop): one stream **per iteration**. The **runtime iteration index enters the
feed's key derivation chain as a split counter** at that position, so iteration *i*'s
stream is simply the one at the realized path with `i` in the iteration slot (e.g.
`[loopSlot, i, feedSlot]`), its key the runtime master folded along that full path —
deterministic, resumable, and reconstructible offline from the path alone. No per-iteration
key storage exists (the derivation is one extra bijection per iteration, negligible next to
the draw itself), and the streams need no enumeration up front. This works identically
whether the loop survives to runtime or is unrolled at concretization (each copy resolves
to the very same key, bit-for-bit).

## Per-stream overrides

`config.Override(RngCollection.Params, [1, 1], seed)` returns a **copy** of the config with
a single stream pinned — addressed by its consumer's ModelId path, as listed by the stream
report — to an explicit seed, replacing the fully folded key for that stream only. Configs
are immutable values: the receiver (including `RngConfig.Default`) is never changed, so
chain `Override` calls to stack pins and keep the result. Because the override replaces the
*result* of the fold, it survives a later `MasterSeed` change. Matching is exact and
per-collection.

```csharp
var config = new RngConfig { MasterSeed = 42 }
    .Override(RngCollection.Params, [1, 1], 1234)
    .Override(RngCollection.Runtime, [2, 0, 1], 5678);
```

Every stream is a valid override address, including the per-iteration streams of a loop
feed (e.g. `[1, 2, 1]` = iteration 2 of the feed at loop slot 1) — overriding one iteration
re-seeds that iteration only; sibling iterations keep their derived keys. An override that
addresses no stream throws: a `Runtime` override at bind (`ApplyRngConfig`), a `Params`
override at parameter initialization. A loop feed's override addresses the site pattern —
static slots exactly, the iteration slot by index — and routes structurally in the feed's
derivation chain, so it works for any iteration the loop actually executes.

Note that changing an override's **value** (or the master seed) on an already-built model
is a pure parameter write, while changing the override **set** re-wires the derivation
chains — fine on an in-memory model, refused (loudly) on a loaded one, whose chains are
baked.

## The stream report

`arch.GetRngStreamReport(config)` inventories every stream of a concrete architecture — the
init stream of each parameter (ModelId path, name, shape, resolved key) and one row per
runtime feed site (path with `-1` marking an iteration slot; static sites carry their exact
resolved key, in-loop sites derive per-iteration keys at runtime) — and can emit the sparse
`Rng.Pin` skeleton for freezing streams before a refactor (see
[Pinning RNG streams](rng-pinning.md)).

## Without a config

There is no unkeyed concrete model. A model concretized without a config gets the **default
deterministic identity** — `RngConfig.Default`, master seed 0 — for both collections:
parameters initialize under it and feeds draw keyed Threefry under it, exactly as if you had
passed it explicitly. "No config" configures the seed to 0; it never falls back to backend
random ops. In particular, repeated one-shot inference of a no-config model repeats its
draws bit-for-bit; if you want per-run variation, say so with `RngConfig.NonDeterministic()`.

There is likewise no per-site seed: `Globals.RandomUniform` / `RandomNormal` take no seed
parameter. Randomness is configuration — a model definition never contains a seed — so all
seeding goes through `RngConfig`, addressed by ModelId when a single stream needs pinning.

## Choosing seeds

- `RngConfig.Default` — master seed 0; fully deterministic, and what "no config" means.
- `new RngConfig { MasterSeed = s }` — deterministic under your seed.
- `RngConfig.NonDeterministic()` — a fresh master from system entropy each run; the chosen
  seed is fixed on the object so the run stays internally consistent and can be recorded.
- `SharedKey = true` — every init stream shares one key, so same-shape parameters receive
  identical values; a debugging/reference-test mode, not for training.
