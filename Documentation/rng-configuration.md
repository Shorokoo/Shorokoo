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

## One seed, per-position streams

Everything derives from `MasterSeed`. Each random consumer — every parameter's
initialization and every runtime feed — draws from its own **stream**, and a stream is fixed
by *where the consumer sits in the model* (its ModelId path), not by draw order. Parameter
initialization and runtime feeds come from two independent collections
(`RngCollection.Params` and `RngCollection.Runtime`), so initialization and per-step
randomness are separately controllable.

Keying by position rather than order is what gives you:

- **Decorrelated streams for free** — two same-shape parameters initialized the same way get
  different values.
- **Reproducibility** — a fixed config always produces the same draws, on any backend.
- **Locality** — inserting or reordering consumers only re-keys the streams whose positions
  move; everything else is undisturbed. [`Rng.Pin`](rng-pinning.md) freezes positions against
  refactoring.
- **Coherent re-seeding** — changing `MasterSeed` re-randomizes everything at once.

## Re-binding, save/load, and training

**Re-binding re-seeds in place.** You can change the seed *after* `ToConcreteModel`, on the
already-built model — randomness re-derives, trained or loaded weights are untouched:

```csharp
var concrete = arch.ToConcreteModel(new RngConfig { MasterSeed = 1 });
// ... later: same model object, different randomness — exact, not approximate.
concrete.ApplyRngConfig(new RngConfig { MasterSeed = 2 });
```

Re-applying the original config restores the original draws bit-for-bit.

**Randomness survives save/load.** A saved model carries its own RNG identity, so a loaded
model draws exactly what it drew before saving — no config object needed on the loading side.
A loaded model is re-bindable like any concrete model. The one exception is a re-imported
*exported ONNX* file (`.onnx`): export bakes the model's RNG keys into plain constants, so
binding a config to a re-import fails loudly rather than silently record a seed the baked
draws would not use.

**Training needs nothing special.** The rig binds randomness at the same point inference does,
before loss and autodiff, so a Dropout's backward mask matches its forward mask by
construction.

## Exported ONNX is deterministic and portable

An exported model computes its randomness from ordinary integer/float ops that call a named,
versioned RNG function — never a backend random op. So an exported model draws the same values
on any execution provider, and the randomness is identifiable: you can point at the function
in the ONNX file that produced a given draw.

## Choosing the generator

`RngConfig.Algorithm` selects the bit generator:

- `RngAlgorithm.Threefry2x32` (default) — Threefry-2x32, 20 rounds; the Random123
  safety-margin default.
- `RngAlgorithm.Threefry2x32Rounds13` — the reduced 13-round variant (Random123
  `threefry2x32x13`): still BigCrush-resistant, ~35% cheaper — the faster, lower-margin choice.

Switching `Algorithm` changes only the numbers drawn, never which stream is which — every
generator keys streams the same way. Both parameter initialization and runtime feeds honor the
choice, and you can switch generators on an already-built model the same way you re-seed it
(`concrete.ApplyRngConfig(...)`).

## Per-step variation

Runtime feeds vary per execution automatically: under the training rig, Dropout masks differ
each step and a resumed run continues where it left off, while one-shot inference keeps its
draws fixed and stays deterministic. Modules never manage any of this — `Globals.RandomUniform`
is all a consumer writes.

## Feeds inside loops

A feed inside a loop gets one stream **per iteration**: iteration *i* draws from the stream at
its position with `i` in the iteration slot, deterministic and resumable. The set of
per-iteration streams is fixed when the model is made concrete — a concrete model's stream set
is static, exactly like its parameter set — so a feed behaves identically whether the loop runs
as a real loop at runtime or is unrolled: the same iteration resolves to the same draw either
way.

## Per-stream overrides

`config.Override(...)` returns a **copy** of the config with a single stream pinned to an
explicit seed — addressed by its consumer's ModelId path, as listed by the stream report.
Configs are immutable values: the receiver (including `RngConfig.Default`) is never changed, so
chain `Override` calls to stack pins. An override replaces the derived key, so it survives a
later `MasterSeed` change, and matching is exact and per-collection.

```csharp
var config = new RngConfig { MasterSeed = 42 }
    .Override(RngCollection.Params, [1, 1], 1234)
    .Override(RngCollection.Runtime, [2, 0, 1], 5678);
```

Every path the stream report lists is a valid override address, including the per-iteration
streams of a loop feed (e.g. `[1, 2, 1]` = iteration 2 of the feed at loop slot 1); overriding
one re-seeds that iteration only, and siblings keep their keys. An override that matches no
stream fails loudly rather than being silently inactive: a `Runtime` override at bind
(`ApplyRngConfig`), a `Params` override at parameter initialization.

Stream addresses are fixed when the model is made concrete from the sample inputs, so — like
trainable params inside loops — a loop feed's streams are enumerated for the hinted trip count.
Driving a loop past that enumerated space is invalid use of the concrete model.

## The stream report

`arch.GetRngStreamReport(config)` inventories every stream of a concrete model — each
parameter's init stream (path, name, shape, resolved key) and every runtime stream (path,
kind, per-iteration key) — and can emit the sparse `Rng.Pin` skeleton for freezing streams
before a refactor (see [Pinning RNG streams](rng-pinning.md)).

## Without a config

There is no unkeyed model. A model made concrete without a config gets the **default
deterministic identity** — `RngConfig.Default`, master seed 0 — for both collections, exactly
as if you had passed it explicitly. "No config" means seed 0; it never falls back to backend
random ops. Repeated one-shot inference of a no-config model repeats its draws bit-for-bit; for
per-run variation, say so with `RngConfig.NonDeterministic()`.

There is likewise no per-site seed: `Globals.RandomUniform` / `RandomNormal` take no seed
parameter. All seeding goes through `RngConfig`, addressed by ModelId path when a single stream
needs pinning.

## Choosing seeds

- `RngConfig.Default` — master seed 0; fully deterministic, and what "no config" means.
- `new RngConfig { MasterSeed = s }` — deterministic under your seed.
- `RngConfig.NonDeterministic()` — a fresh master from system entropy each run; the chosen
  seed is fixed on the object so the run stays internally consistent and can be recorded.
