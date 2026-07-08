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

## Binding writes the model's RNG identity

Applying a config does **not** transform the graph. Binding validates the config against the
model's stream inventory and writes one thing: a compact **key vector** the model carries as
a single parameter-like tensor — its RNG identity (master seed, sub-masters, any per-stream
overrides, and the algorithm name). No feed is touched, no opcode changes, no inputs change.
Structure is deferred: only when a graph is prepared for execution or ONNX export does the
lowering derive each feed's keys from that identity and rewrite the feed into the keyed
deterministic draw — and that rewrite happens on the export-time clone, never on your graph.

Three properties follow directly:

**Re-binding is cheap and late.** Because binding is replacing one tensor, you can change
the master seed *after* `ToConcreteModel`, on the already-built model:

```csharp
var concrete = arch.ToConcreteModel(new RngConfig { MasterSeed = 1 });
// ... later: same model object, different randomness — exact, not approximate.
concrete.ApplyRngConfig(new RngConfig { MasterSeed = 2 });
```

Re-applying the original config restores the original draws bit-for-bit.

**Randomness survives save/load.** The identity tensor rides the model file (as a
reserved-name initializer), so a loaded model draws exactly what it drew before saving — no
config object needed on the loading side.

**Training needs nothing special.** The rig binds the concrete architecture at the same
shared point inference uses, *before* loss composition and autodiff. Since binding changes
no opcodes, autodiff sees exactly the graph it always saw; the identity tensor rides along
into the training-step graph, and the forward and any recomputed backward copy of a feed
derive the same key — so e.g. a Dropout's backward mask matches its forward mask by
construction.

## What a keyed feed lowers to

At ONNX prep, a feed of a bound model becomes a call to the config's **named RNG
algorithm** — a versioned set of functions (kinds *split* / *uniform* / *normal*) that
export as tagged, non-inlined ONNX local `FunctionProto`s. The exported model's randomness
is therefore deterministic, portable across execution providers, and identifiable: you can
point at the function in the ONNX file that produced any draw.

## Choosing the generator

`RngConfig.Algorithm` selects the bit generator:

- `RngAlgorithm.Threefry2x32` (default) — Threefry-2x32, 20 rounds; the Random123
  safety-margin default.
- `RngAlgorithm.Threefry2x32Rounds13` — the reduced 13-round variant (Random123
  `threefry2x32x13`): still BigCrush-resistant, ~35% cheaper — the faster, lower-margin choice.

All algorithms **share one key tree**: a stream's key is derived the same way regardless of
generator, so switching `Algorithm` never reshuffles which stream is which — the same stream
simply draws different numbers. Both parameter initialization and runtime feeds honor the
choice. Because binding is replacing the identity tensor, you can switch generators on an
already-built model the same way you re-seed it (`concrete.ApplyRngConfig(...)` /
re-materialize for init), and the exported model calls — and is tagged with — the selected
algorithm's functions.

Per-execution variation is carried by a separate **drawBase** counter, not by the key — and
the RNG system manages it itself: concretization injects one model-global execution counter
(`RngExecutionCounter`, ordinary model state, initialized 0 and advanced +1 per execution)
and wires it into every feed. Modules never touch it — `Globals.RandomUniform` is all a
consumer writes. Under the training rig the counter rides the checkpoint, so Dropout masks
differ per step and a resumed run at step N draws exactly what the uninterrupted run would;
in one-shot inference it is baked at 0, so inference stays deterministic and stateless. One
counter serves all feeds because sites are already decorrelated by their stream keys; it
costs the checkpoint a single scalar.

## Feeds inside loops

A feed under a loop takes a ModelId with a `-1` iteration slot (exactly like a parameter
created in a loop): one stream **per iteration**. Its per-iteration streams are enumerated
at `ToConcreteArchitecture` — a concrete architecture's stream set is static, exactly like
its parameter set — and at ONNX prep the feed draws from a per-stream **key table** with
the row selected by the runtime iteration index, so iteration *i* draws from
`fold(fold(prefixKey, i), …)` — deterministic, resumable, and reconstructible offline from
the path and the iteration number. This works identically whether the loop survives to
runtime (an ONNX `Loop` selecting a row per iteration) or is unrolled at concretization
(each copy resolves to the very same key, bit-for-bit).

## Per-stream overrides

`config.Override(RngCollection.Params, [1, 1], seed)` pins a single stream — addressed by
its consumer's ModelId path, as listed by the stream report — to an explicit seed,
replacing the fully folded key for that stream only. Because the override replaces the
*result* of the fold, it survives a later `MasterSeed` change. Matching is exact and
per-collection.

Every path the stream report lists is a valid override address, including the realized
per-iteration streams of a loop feed (e.g. `[1, 2, 1]` = iteration 2 of the feed at loop
slot 1) — overriding one iteration re-seeds that iteration only; sibling iterations keep
their derived keys. An override that matches no stream fails loudly rather than being
silently inactive: a `Runtime` override at bind (`ApplyRngConfig`), a `Params` override at
parameter initialization.

Stream ids are realized at `ToConcreteArchitecture` from the input hints, so — like
trainable params inside loops — a loop feed's streams are enumerated for the hinted trip
count: the concrete architecture is only valid for inputs that produce the same model-id
list (or a subset). Driving a loop past its enumerated iteration space is invalid use of
the concrete artifact — static ModelIds are what "concrete" means.

## The stream report

`arch.GetRngStreamReport(config)` inventories every stream of a concrete architecture — the
init stream of each parameter (ModelId path, name, shape, resolved key) and every realized
runtime stream (path, kind, exact per-iteration key) — and can emit the sparse `Rng.Pin`
skeleton for freezing streams before a refactor (see
[Pinning RNG streams](rng-pinning.md)).

## Without a config

A model that was never bound carries no RNG identity, and its feeds lower to the plain ONNX
random ops (`RandomUniformLike` / `RandomNormalLike`): the conventional, backend-seeded
behavior — non-reproducible by nature, with any user-supplied seed passed through and none
synthesized.

## Choosing seeds

- `RngConfig.Default` — master seed 0; fully deterministic.
- `new RngConfig { MasterSeed = s }` — deterministic under your seed.
- `RngConfig.NonDeterministic()` — a fresh master from system entropy each run; the chosen
  seed is fixed on the object so the run stays internally consistent and can be recorded.
- `SharedKey = true` — every init stream shares one key, so same-shape parameters receive
  identical values; a debugging/reference-test mode, not for training.
