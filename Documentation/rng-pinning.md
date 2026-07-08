# Pinning RNG streams with `Rng.Pin`

Every random consumer in a module — a sub-model created by `X.Model(...)`, a trainable
parameter created by an initializer `Init(...)` call, a runtime random feed — draws from its
own RNG **stream**. A stream's key is derived from the consumer's ModelId (see
[Configuring randomness](rng-configuration.md)): its position in
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

**The compiler writes the pins for you.** For a `[Module]` whose `Inline` body captures every
`Model(...)` / `Init(...)` result in a local variable, the source generator emits an Info
diagnostic (`MSG004`) carrying the exact ready-to-paste statements, in current creation order
— so freezing such a module is one copy-paste. This covers `LoopAPI.Iterate(...)` loops at
any nesting depth: because a pin reshapes only its own scope, the generator emits **one pin
per scope**, each placed where its variables are in scope — a module-level pin at the end of
`Inline`, and a pin inside each loop body:

```csharp
public static Tensor<float32> Inline(Tensor<float32> x)
{
    var a = Linear.Model(Scalar(2L), Scalar(false));
    x = a.Call(x);
    foreach (var ctx in LoopAPI.Iterate(steps))
    {
        var w = KaimingUniform.Init(shape);
        x = x + w;                        // ... use w ...
        Rng.Pin(w);                       // pins this loop's local slots
        ctx.ContinueWhile(cond);
    }
    var b = Linear.Model(Scalar(3L), Scalar(false));
    x = b.Call(x);
    Rng.Pin(([1], a), ([3], b));          // module-level: loop keeps slot 2
    return x;                             // forward result, returned after the pin
}
```

The generator *refuses* to emit a suggestion for anything it cannot fully analyze in **any**
scope: bodies with C# control flow (`if`/`switch`/raw `for`/`while`/a non-`Iterate`
`foreach`), chained `X.Model(...).Call(...)`, static `X.Call(...)` shortcuts, or opaque helper
calls that may create streams internally. This is by design — a wrong pin silently changes
seeds, so no suggestion beats a bad one.

**The report supplies slots, you supply names.** For bodies the generator refuses, the
bind-time stream report (`arch.GetRngStreamReport(config)`) describes every slot — ModelId
path, consumer kind, parameter name, shape, resolved key — and its `EmitPinSkeleton()` emits
per-scope sparse skeletons with the one thing it cannot know left as a placeholder for you to
fill in. Attaching a durable source identity to a stream is inherently a source-side act: C#
is Turing-complete, a loop makes one identifier refer to many consumers, an `if` makes a
consumer exist conditionally — so any tool that claimed to bind names for you would be
guessing, and a guessed pin is worse than none.

## The sparse form

`Rng.Pin(([3], item), ...)` pins each listed item to exactly the named **local slot in the
scope the pin is written in**. The named slots are *reservations*: unlisted consumers fill
the remaining free slots in creation order. Two consequences:

- Pinning items to their **current** slots — what the stream report's skeleton emits — leaves
  every unlisted consumer's slot, hence stream, unchanged. This is the freeze workflow, and
  the reason the skeleton uses this form.
- Pinning an item to a **different** slot perturbs the free-slot sequence: an unlisted
  consumer whose slot was taken (or vacated) can move and silently re-key — e.g. with `a`
  and `b` at slots 1 and 2, `Rng.Pin(([2], a))` displaces the unlisted `b` to slot 1. To
  relocate streams, list every consumer the move disturbs; an intentional swap is
  `Rng.Pin(([2], a), ([1], b))`.

The path is a single 1-based local slot; the scope is the module body, or the loop body the
pin is written in (to pin a loop's consumer, write the sparse pin inside that loop).

## Current limits

- A consumer whose variable is scoped inside a C# `if`/`for`/`while` block (rather than a
  `LoopAPI.Iterate` loop body, which pins support) cannot be referenced by an end-of-scope
  pin; such consumers remain positional.
- A pin that cannot be resolved — an unsupported item type, a handle created outside the
  module body, or one that leads to no id-bearing node — **fails the module build** with an
  `Rng.Pin` error: an inactive pin the author believes is active is exactly the silent
  re-keying the feature exists to prevent.
