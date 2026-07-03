# Pinning RNG streams with `Rng.Pin`

Every random consumer in a module — a sub-model created by `X.Model(...)`, a trainable
parameter created by an initializer `Init(...)` call, a runtime random feed — draws from its
own RNG **stream**. A stream's key is derived from the consumer's ModelId: its position in
the module's id address space, assigned in creation order. That makes streams reproducible
and decorrelated by default, but **positional**: inserting or reordering siblings inside a
module body shifts the ModelIds of the consumers after the edit point, and their streams
re-key.

`Rng.Pin` freezes stream identity against such refactoring.

## The core principle: the variable name is the durable identity

A computation graph only knows *positions*. Your source code knows *names*. When you move a
line, the one thing that survives the move is the identifier that captured the value —
`var w = KaimingUniform.Init(...)` can go anywhere in the body and `w` is still `w`. `Rng.Pin`
works by extracting exactly that source-level identity and binding it to an id slot:

```csharp
[Module]
public partial class Encoder
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var proj = Linear.Model(Scalar(128L), Scalar(true));
        var head = Linear.Model(Scalar(16L), Scalar(false));
        var y = head.Call(proj.Call(x));
        Rng.Pin(proj, head);      // proj -> slot 1, head -> slot 2 — regardless of code order
        return y;
    }
}
```

With the pin in place, `proj` and `head` keep their streams no matter how the body is
rearranged: the pin list — not creation order — defines the slot assignment. It is the same
reason a rename registers as a *move* rather than a *delete-plus-create* in version control:
something must track identity across positions, and here that something is the variable name
you placed in the pin. **Without the name there is nothing to pin** — no runtime or
graph-side tool can substitute for it, because names exist only in source.

## Contract

- `Rng.Pin(a, b, c)` is called once, at the end of the module's `Inline` body. Listed items
  take the module-local id slots **in list order** (first item = slot 1). Unlisted consumers
  follow, in node order.
- Nothing enters the computation graph: the pin only reshapes how the module compiler numbers
  the graph it was already building.
- **To freeze a module's current streams before a refactor, list ALL of its random consumers
  in current creation order.** A partial positional pin re-keys the unlisted consumers (they
  move behind the pinned ones).
- Pinning also stabilizes the pinned items' identifier names (`Linear#0` vs `Linear#1` follow
  slot order), so checkpoint parameter names for pinned items are refactor-stable too.
- Same model object = same slot = same stream: calling one model object twice draws from one
  stream (the deliberate weight-sharing-like coupling); independent streams need distinct
  objects.

## Tooling, and the division of labor

**Straight-line bodies: the compiler writes the pin for you.** For a `[Module]` whose
`Inline` body has no C# control flow and captures every `Model(...)` / `Init(...)` result in
a local variable, the source generator emits an Info diagnostic (`MSG004`) carrying the exact
ready-to-paste statement, in current creation order — so freezing such a module is one
copy-paste. The generator *refuses* to emit a suggestion for anything it cannot fully
analyze: bodies with `if`/loops, chained `X.Model(...).Call(...)`, static `X.Call(...)`
shortcuts, or opaque helper calls that may create streams internally. This is by design — a
wrong pin silently changes seeds, so no suggestion beats a bad one.

**Everything else: tools supply slots, you supply names.** For bodies the generator refuses,
a bind-time stream report can describe every slot — ModelId path, consumer kind, parameter
name, shape — and emit a skeleton for the sparse form with the one thing it cannot know left
as a placeholder:

```csharp
Rng.Pin(
    ([1], /* Linear#0.weight  [64,128] */ ?),
    ([2], /* Dropout#0 mask feed       */ ?),
    ([4,2], /* loop-body Linear.weight */ ?));
```

Filling in the `?`s is the author's job, necessarily: attaching a durable source identity to
a stream is inherently a source-side act. No automatic linkage is possible in general — C#
is Turing-complete, a `for` loop makes one identifier refer to many consumers, an `if` makes
a consumer exist conditionally — so any tool that claimed to bind names for you in such
bodies would be guessing, and a guessed pin is worse than none.

## Current limits

- The positional form is implemented. The sparse explicit-index form
  (`Rng.Pin(([3], item), ...)`, module-local id paths with loop-iteration slots elided) is
  specified but not yet implemented; it makes *partial* pinning seed-preserving (pin items at
  their current slots; unpinned items keep theirs) and is the form tools should emit.
- A consumer whose variable is scoped inside a C# block (`if`/`for`) cannot be referenced by
  an end-of-body pin at all; such consumers remain positional.
- A pinned handle that cannot be resolved to a graph node is currently ignored silently;
  this will become a build error (an inactive pin the author believes is active is exactly
  the silent re-keying the feature exists to prevent).
