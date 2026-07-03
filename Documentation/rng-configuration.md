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

## Binding is stamping, not rewriting

Applying a config does **not** transform the graph. The binding pass resolves each runtime
feed's stream key host-side and stamps it as an *attribute* on the feed's own node
(`shrk_rng_explicit_key`, the two 32-bit key words, next to the algorithm name). No opcode
changes, no inputs change, no nodes are added. Structure is deferred: only when a graph is
prepared for execution or ONNX export does the lowering read the stamp and rewrite the feed
into the keyed deterministic draw — and that rewrite happens on the export-time clone, never
on your graph.

Two properties follow directly:

**Re-binding is cheap and late.** Because the pre-export graph keeps its shape, you can
change the master seed *after* `ToConcreteModel`, on the already-built model, and a re-stamp
simply overwrites the previous stamp:

```csharp
var concrete = arch.ToConcreteModel(new RngConfig { MasterSeed = 1 });
// ... later: same model object, different randomness — exact, not approximate.
concrete.ApplyRngConfig(new RngConfig { MasterSeed = 2 });
```

Re-applying the original config restores the original draws bit-for-bit.

**Training needs nothing special.** The rig stamps the concrete architecture at the same
shared point inference uses, *before* loss composition and autodiff. Since stamping changes
no opcodes, autodiff sees exactly the graph it always saw; the stamped attributes ride along
into the training-step graph, and the forward and any recomputed backward copy of a feed
carry the same key — so e.g. a Dropout's backward mask matches its forward mask by
construction.

## What a stamped feed lowers to

At ONNX prep, a stamped feed becomes a call to the config's **named RNG algorithm** — a
versioned set of functions (default `"Threefry2x32-BoxMuller.v1"`, kinds *split* / *uniform*
/ *normal*) that export as tagged, non-inlined ONNX local `FunctionProto`s. The exported
model's randomness is therefore deterministic, portable across execution providers, and
identifiable: you can point at the function in the ONNX file that produced any draw.

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
created in a loop): conceptually one stream **per iteration**. The stamp carries the key of
the path prefix before the `-1`; at ONNX prep the remaining slots become in-graph `split`
folds on the runtime iteration index, so iteration *i* draws from
`fold(fold(prefixKey, i), …)` — deterministic, resumable, and reconstructible offline from
the path and the iteration number. This works identically whether the loop survives to
runtime (an ONNX `Loop` splitting per iteration) or is unrolled at concretization (the
splits constant-fold to the very same keys, bit-for-bit).

## Per-stream overrides

`config.Override(RngCollection.Params, [1, 1], seed)` pins a single stream — addressed by
its consumer's ModelId path, as listed by the stream report — to an explicit seed,
replacing the fully folded key for that stream only. Because the override replaces the
*result* of the fold, it survives a later `MasterSeed` change. Matching is exact and
per-collection.

## The stream report

`arch.GetRngStreamReport(config)` inventories every stream of a concrete architecture — the
init stream of each parameter (ModelId path, name, shape, resolved key) and each runtime
feed (path, kind; for loop-body feeds the prefix key) — and can emit the sparse `Rng.Pin`
skeleton for freezing streams before a refactor (see
[Pinning RNG streams](rng-pinning.md)).

## Without a config

A feed with no stamp lowers to the plain ONNX random ops (`RandomUniformLike` /
`RandomNormalLike`): the conventional, backend-seeded behavior — non-reproducible by nature,
with any user-supplied seed passed through and none synthesized.

## Choosing seeds

- `RngConfig.Default` — master seed 0; fully deterministic.
- `new RngConfig { MasterSeed = s }` — deterministic under your seed.
- `RngConfig.NonDeterministic()` — a fresh master from system entropy each run; the chosen
  seed is fixed on the object so the run stays internally consistent and can be recorded.
- `SharedKey = true` — every init stream shares one key, so same-shape parameters receive
  identical values; a debugging/reference-test mode, not for training.
