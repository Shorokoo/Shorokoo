# Shorokoo.CodeGen

C# source generator for [Shorokoo](https://github.com/Shorokoo/Shorokoo).

Turns `[Module]`-annotated partial classes into strongly typed `Module`/`Model`
wrappers — generated `Model()` / `Call()` factory methods, cached computation
graphs, and typed hyperparameter sets for optimizers.

```bash
dotnet add package Shorokoo.CodeGen
```

This is a development-time dependency (a Roslyn analyzer); it adds nothing to
your runtime output. Shorokoo is usable without it, but `[Module]` classes are
the recommended way to define models.

Documentation: https://github.com/Shorokoo/Shorokoo
