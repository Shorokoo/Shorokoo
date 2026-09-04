# Simple Pattern Matching DSL

Related: [onnx-and-weights.md](onnx-and-weights.md) ·
[param-naming-format-dsl.md](param-naming-format-dsl.md)

Reference for the patterns accepted by `SimplePatternScheme` /
`SimplePatternNamingScheme` — one of the two ways to map parameter names when
[binding third-party weights](onnx-and-weights.md#bind-loaded-weights-into-a-model-for-inference)
(e.g. PyTorch/timm checkpoints) into a model with
`ToConcreteModel(weights, namingScheme)`. The alternative, formatting
ModelIds positionally, is the
[ModelId format DSL](param-naming-format-dsl.md).

A custom pattern language for converting Shorokoo IDs to third-party framework parameter names. Designed for simplicity—no regex required.

## 1. Semantic Elements

A Shorokoo ID is parsed into **semantic elements**—the fundamental units of matching. Each element is one of:

| Element Type | Description | Examples |
|--------------|-------------|----------|
| **Word** | Contiguous letters | `Conv`, `BatchNorm`, `layer` |
| **Number** | Contiguous digits (as a unit) | `77`, `22`, `0`, `123` |
| **Dot** | The `.` character | `.` |
| **Hash** | The `#` character | `#` |
| **Colon** | The `:` character | `:` |

**Parsing rules:**
- Letters and digits form separate elements at boundaries
- `.`, `#` and `:` are always individual elements
- Numbers are treated as single elements regardless of digit count
- Any other character is silently skipped: it yields no element, but it still
  ends the run of letters or digits before it — `Conv-x` parses as `Conv`, `x`,
  not as one word

**Example:**

```
Input: "Conv2Dk77s22#0.BatchNorm#1"

Semantic elements:
  [0]  "Conv"      (word)
  [1]  "2"         (number)
  [2]  "Dk"        (word)
  [3]  "77"        (number)
  [4]  "s"         (word)
  [5]  "22"        (number)
  [6]  "#"         (hash)
  [7]  "0"         (number)
  [8]  "."         (dot)
  [9]  "BatchNorm" (word)
  [10] "#"         (hash)
  [11] "1"         (number)
```

A loop index parses the same way, the `:` being an element of its own:

```
Input: "Loop#0:12"

Semantic elements:
  [0] "Loop"      (word)
  [1] "#"         (hash)
  [2] "0"         (number)
  [3] ":"         (colon)
  [4] "12"        (number)
```

## 2. Escape Sequences

| Syntax | Produces | Use Case |
|--------|----------|----------|
| `\o` | `{` | Literal opening brace |
| `\s` | `\` | Literal backslash |

Note: Closing brace `}` does not require escaping.

## 3. Pattern Syntax

### 3.1 Literals

Match text exactly as written:

```
Pattern: "BatchNorm#0"
Matches: "BatchNorm#0"
```

### 3.2 Wildcards

| Syntax | Description |
|--------|-------------|
| `{*}` | Match any run of elements, including an empty one |

```
Pattern: "Layer#{*}.weight"
Matches: "Layer#0.weight", "Layer#123.weight", "Layer#0.Sub#1.weight"
```

The matcher backtracks and takes the **shortest** run that lets the rest of the
pattern match — so where a pattern holds two `{*}`, the first one takes as few
elements as it can.

### 3.3 Captures

| Syntax | Description |
|--------|-------------|
| `{name}` | Capture 1 semantic element |
| `{name:n}` | Capture n semantic elements |

**Single element capture:**

```
Pattern: "Loop#0:{idx}"
Input:   "Loop#0:5"
Result:  idx = "5"
```

**Multi-element capture:**

```
Pattern: "Loop#0:{idx}.{mod:2}#0"
Input:   "Loop#0:3.Conv2#0"
Result:  idx = "3", mod = "Conv2" (2 elements: "Conv" + "2")
```

### 3.4 Range Constraints

Constrain numeric captures to specific values:

| Syntax | Matches |
|--------|---------|
| `{n\|1:3}` | 1, 2, 3 (inclusive range) |
| `{n\|2:}` | 2, 3, 4, ... (open-ended) |
| `{n\|:5}` | 0, 1, 2, 3, 4, 5 (from 0) |
| `{n\|1::2}` | 1, 3, 5, 7, ... (start:end:step, odd numbers) |
| `{n\|0::2}` | 0, 2, 4, 6, ... (even numbers) |

**Examples:**

```
Pattern: "Loop#0:{idx|1:3}"
Matches: "Loop#0:1", "Loop#0:2", "Loop#0:3"
Rejects: "Loop#0:0", "Loop#0:4"

Pattern: "Block#{b|1::2}"
Matches: "Block#1", "Block#3", "Block#5" (odd only)
```

## 4. Format String Syntax

### 4.1 Placeholders

| Syntax | Description | Example |
|--------|-------------|---------|
| `{name}` | Insert captured value | `{idx}` → "2" |
| `{name + N}` | Add offset | `{idx + 1}` → "3" |
| `{name - N}` | Subtract offset | `{idx - 1}` → "1" |

### 4.2 Map Lookups

| Syntax | Description |
|--------|-------------|
| `{name\|mapName}` | Lookup in named map |
| `{name\|lower}` | Convert to lowercase |

```csharp
maps: new() { ["bnParam"] = new() { ["0"] = "running_mean", ["1"] = "running_var" } }
format: "{p|bnParam}"  // p=0 → "running_mean"
```

## 5. Complete ResNet50 Example

### 5.1 Scheme Definition

The patterns go into a `SimplePatternNamingScheme`, which also takes the
model's own canonical-id scheme (`arch.GetShorokooIdNamingScheme()`, from the
concrete architecture) and a framework id. Every pattern whose format reaches
for a map is handed the maps of [§5.2](#52-shared-maps) — a `SimplePatternScheme`
looks only in its own maps, so one that omits them throws on `{p|bnParam}`.

```csharp
public static SimplePatternNamingScheme CreateResNet50Scheme(ModelIdNamingScheme shorokooIdScheme)
{
    SimplePatternScheme[] patterns =
    [
        // ════════════════════════════════════════════════════════════════
        // STEM
        // ════════════════════════════════════════════════════════════════
        new SimplePatternScheme(
            pattern: "ResNetStem#0.Conv2Dk77s22#0.InitSimple#0",
            format:  "conv1.weight"
        ),
        new SimplePatternScheme(
            pattern: "ResNetStem#0.BatchNorm#0.InitSimple#{p}",
            format:  "bn1.{p|bnParam}",
            maps:    SharedMaps
        ),

        // ════════════════════════════════════════════════════════════════
        // LAYER 1 - First block (with downsample)
        // ════════════════════════════════════════════════════════════════
        new SimplePatternScheme(
            pattern: "BottleneckStackS11#0.Loop#0:0.BottleneckS11#0.Conv2Dk11s11#{c}.InitSimple#0",
            format:  "layer1.0.{c|layer1Conv}.weight",
            maps:    SharedMaps
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS11#0.Loop#0:0.BottleneckS11#0.Conv2Dk33s11#0.InitSimple#0",
            format:  "layer1.0.conv2.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS11#0.Loop#0:0.BottleneckS11#0.BatchNorm#{b}.InitSimple#{p}",
            format:  "layer1.0.{b|layer1Bn}.{p|bnParam}",
            maps:    SharedMaps
        ),

        // ════════════════════════════════════════════════════════════════
        // LAYER 1 - Remaining blocks (idx >= 1, no downsample)
        // ════════════════════════════════════════════════════════════════
        new SimplePatternScheme(
            pattern: "BottleneckStackS11#0.Loop#0:{idx|1:}.BottleneckS11#0.Conv2Dk11s11#0.InitSimple#0",
            format:  "layer1.{idx}.conv1.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS11#0.Loop#0:{idx|1:}.BottleneckS11#0.Conv2Dk33s11#0.InitSimple#0",
            format:  "layer1.{idx}.conv2.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS11#0.Loop#0:{idx|1:}.BottleneckS11#0.Conv2Dk11s11#1.InitSimple#0",
            format:  "layer1.{idx}.conv3.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS11#0.Loop#0:{idx|1:}.BottleneckS11#0.BatchNorm#{b}.InitSimple#{p}",
            format:  "layer1.{idx}.bn{b + 1}.{p|bnParam}",
            maps:    SharedMaps
        ),

        // ════════════════════════════════════════════════════════════════
        // LAYERS 2-4 - Generalized
        // BottleneckStackS22#0 → layer2, #1 → layer3, #2 → layer4
        // ════════════════════════════════════════════════════════════════

        // First block: BottleneckS22 with stride-2 and downsample
        new SimplePatternScheme(
            pattern: "BottleneckStackS22#{layer}.Loop#0:0.BottleneckS22#0.Conv2Dk11s11#0.InitSimple#0",
            format:  "layer{layer + 2}.0.conv1.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS22#{layer}.Loop#0:0.BottleneckS22#0.Conv2Dk33s22#0.InitSimple#0",
            format:  "layer{layer + 2}.0.conv2.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS22#{layer}.Loop#0:0.BottleneckS22#0.Conv2Dk11s11#1.InitSimple#0",
            format:  "layer{layer + 2}.0.conv3.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS22#{layer}.Loop#0:0.BottleneckS22#0.BatchNorm#{b}.InitSimple#{p}",
            format:  "layer{layer + 2}.0.{b|bnDs}.{p|bnParam}",
            maps:    SharedMaps
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS22#{layer}.Loop#0:0.BottleneckS22#0.Conv2Dk11s22#0.InitSimple#0",
            format:  "layer{layer + 2}.0.downsample.0.weight"
        ),

        // Remaining blocks: BottleneckS11 (idx >= 1, no downsample)
        new SimplePatternScheme(
            pattern: "BottleneckStackS22#{layer}.Loop#0:{idx|1:}.BottleneckS11#0.Conv2Dk11s11#0.InitSimple#0",
            format:  "layer{layer + 2}.{idx}.conv1.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS22#{layer}.Loop#0:{idx|1:}.BottleneckS11#0.Conv2Dk33s11#0.InitSimple#0",
            format:  "layer{layer + 2}.{idx}.conv2.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS22#{layer}.Loop#0:{idx|1:}.BottleneckS11#0.Conv2Dk11s11#1.InitSimple#0",
            format:  "layer{layer + 2}.{idx}.conv3.weight"
        ),
        new SimplePatternScheme(
            pattern: "BottleneckStackS22#{layer}.Loop#0:{idx|1:}.BottleneckS11#0.BatchNorm#{b}.InitSimple#{p}",
            format:  "layer{layer + 2}.{idx}.bn{b + 1}.{p|bnParam}",
            maps:    SharedMaps
        ),

        // ════════════════════════════════════════════════════════════════
        // CLASSIFICATION HEAD
        // ════════════════════════════════════════════════════════════════
        new SimplePatternScheme(
            pattern: "ClassificationHead#0.DenseBasic#0.InitSimple#{p}",
            format:  "fc.{p|fcParam}",
            maps:    SharedMaps
        )
    ];

    return new SimplePatternNamingScheme(
        patterns, shorokooIdScheme, ModuleParamSetNamingScheme.PyTorchFrameworkId);
}
```

### 5.2 Shared Maps

```csharp
static readonly Dictionary<string, Dictionary<string, string>> SharedMaps = new()
{
    ["bnParam"]    = new() { ["0"] = "running_mean", ["1"] = "running_var", ["2"] = "weight", ["3"] = "bias" },
    ["layer1Conv"] = new() { ["0"] = "conv1", ["1"] = "conv3", ["2"] = "downsample.0" },
    ["layer1Bn"]   = new() { ["0"] = "bn1", ["1"] = "bn2", ["2"] = "bn3", ["3"] = "downsample.1" },
    ["bnDs"]       = new() { ["0"] = "bn1", ["1"] = "bn2", ["2"] = "bn3", ["3"] = "downsample.1" },
    ["fcParam"]    = new() { ["0"] = "weight", ["1"] = "bias" }
};
```

### 5.3 Example Conversions

What `CreateResNet50Scheme(...).ToName(shorokooId)` returns for a sample of the
parameters:

| Shorokoo ID | PyTorch Name |
|-------------|--------------|
| `ResNetStem#0.Conv2Dk77s22#0.InitSimple#0` | `conv1.weight` |
| `ResNetStem#0.BatchNorm#0.InitSimple#2` | `bn1.weight` |
| `BottleneckStackS11#0.Loop#0:0.BottleneckS11#0.Conv2Dk11s11#0.InitSimple#0` | `layer1.0.conv1.weight` |
| `BottleneckStackS11#0.Loop#0:1.BottleneckS11#0.BatchNorm#1.InitSimple#3` | `layer1.1.bn2.bias` |
| `BottleneckStackS22#0.Loop#0:0.BottleneckS22#0.Conv2Dk33s22#0.InitSimple#0` | `layer2.0.conv2.weight` |
| `BottleneckStackS22#0.Loop#0:2.BottleneckS11#0.BatchNorm#0.InitSimple#0` | `layer2.2.bn1.running_mean` |
| `BottleneckStackS22#1.Loop#0:3.BottleneckS11#0.Conv2Dk11s11#1.InitSimple#0` | `layer3.3.conv3.weight` |
| `BottleneckStackS22#2.Loop#0:0.BottleneckS22#0.Conv2Dk11s22#0.InitSimple#0` | `layer4.0.downsample.0.weight` |
| `ClassificationHead#0.DenseBasic#0.InitSimple#0` | `fc.weight` |

## 6. API Reference

### `SimplePatternScheme` — one pattern

```csharp
public class SimplePatternScheme
{
    public SimplePatternScheme(
        string pattern,
        string format,
        Dictionary<string, Dictionary<string, string>>? maps = null
    );

    public string Pattern { get; }
    public string Format { get; }
    public ImmutableDictionary<string, ImmutableDictionary<string, string>> Maps { get; }

    public bool Matches(string shorokooId);
    public bool TryMatch(string shorokooId, out Dictionary<string, string> captures);
    public string ToName(string shorokooId);

    public static List<SemanticElement> ParseSemanticElements(string input);
}
```

`TryMatch` is `Matches` plus the bindings: on a match it hands back the capture table the
format string would be evaluated against (`["idx"] = "3"`, `["mod"] = "Conv2"`, as in
[§3.3](#33-captures)), which is how to see what a pattern actually bound without writing a
format for it. `ParseSemanticElements` is the [§1](#1-semantic-elements) parser itself — call
it on an id to see the element list a pattern is matched against:

```csharp
foreach (var e in SimplePatternScheme.ParseSemanticElements("Loop#0:12"))
    Console.WriteLine($"{e.Type} {e.Value}");   // Word Loop / Hash # / Number 0 / Colon : / Number 12
```

### `SimplePatternNamingScheme` — the whole scheme

```csharp
public class SimplePatternNamingScheme : ModuleParamSetNamingScheme
{
    // modelIdToShorokooIdScheme is the model's own canonical-id scheme
    // (arch.GetShorokooIdNamingScheme(), §5.1); frameworkId records which framework's
    // names this scheme speaks, e.g. ModuleParamSetNamingScheme.PyTorchFrameworkId.
    public SimplePatternNamingScheme(
        IEnumerable<SimplePatternScheme> patterns,
        ModelIdNamingScheme modelIdToShorokooIdScheme,
        string frameworkId
    );

    public ImmutableArray<SimplePatternScheme> Patterns { get; }
    public ModelIdNamingScheme ModelIdToShorokooIdScheme { get; }
    public string FrameworkId { get; }                  // from ModuleParamSetNamingScheme

    public override string? ToName(string shorokooId);
    public override string? ToName(ModelId modelId);
    public override string? ToName(ConcreteModelParamInfo shorokooParam);
    public override ModelId? ToModelId(string paramName, ImmutableArray<ModelId> candidates);
}
```

Every `ToName` overload does the same thing — try the patterns in order, return the first
match's name, return `null` when none matches — and they differ only in where the canonical
id string comes from. `ToName(string)` is handed one directly, which is the natural direction
here and the one weight **export** needs. `ToName(ModelId)` gets there through
`ModelIdToShorokooIdScheme` first; `ToName(ConcreteModelParamInfo)` reads the parameter's own
id string off the concrete model.

`ToModelId` is the reverse direction, and the one
`ToConcreteModel(weights, namingScheme)` uses to **import**: it names every candidate ModelId
once into a name → ModelId table, then looks the third-party name up in it. See
[§7](#7-error-handling) — the lookup returns `null` for an unknown name, but building the
table over an incompletely covered set of candidates throws.

## 7. Error Handling

There is no DSL-specific exception type; failures surface as the ordinary BCL
ones — and the most common failure does not throw at all.

| Failure | Behaviour |
|---------|-----------|
| No pattern in a `SimplePatternNamingScheme` matches the id | `ToName` returns **`null`** |
| A single `SimplePatternScheme.ToName` is given an id its pattern does not match | `InvalidOperationException` |
| The format looks up a key the map does not hold | `KeyNotFoundException` |
| The format names a map the scheme was not given | `KeyNotFoundException` |
| The format references a capture the pattern does not bind | `KeyNotFoundException` |
| A pattern or format has an unmatched `{`, or an unparsable range constraint | `FormatException` |
| `ToModelId` is given candidates the patterns do not name in full | `ArgumentNullException` — "Value cannot be null. (Parameter 'key')" |
| `ToModelId` is given two candidates the patterns give the same name | `ArgumentException` — "An item with the same key has already been added" |

```csharp
// No pattern matches: null, not an exception.
string? name = scheme.ToName(unknownId);   // null — nothing named this id

// A lone pattern, on the other hand, insists on matching.
try { var n = new SimplePatternScheme("A#0", "x").ToName("B#1"); }
catch (InvalidOperationException) { /* "Shorokoo ID 'B#1' does not match pattern 'A#0'" */ }

// Map miss — including a scheme constructed without its maps at all.
try { var n = pattern.ToName(id); }
catch (KeyNotFoundException) { /* "Key '9' not found in map 'bnParam'" */ }

// The reverse direction is not forgiving in the same way: ToModelId names every
// candidate before it looks anything up, and an unnamed one lands as a null key.
try { var id2 = partialScheme.ToModelId("layer1.0.conv1.weight", candidates); }
catch (ArgumentNullException) { /* "Value cannot be null. (Parameter 'key')" */ }
```

The two `ToModelId` rows are worth reading twice, because the entry point this page
recommends — `ToConcreteModel(weights, namingScheme)` — goes through them. The lookup
itself is as forgiving as `ToName`: an unknown third-party name gives `null` back. The
table it looks in is not. That table is built over *every* candidate first, so a single
parameter the patterns leave unnamed puts a `null` key into it, and the failure arrives
as a bare `ArgumentNullException` naming neither the parameter nor the scheme. Do not
read it as a complaint about the name you passed; it means the pattern set has a hole
somewhere else in the model. Prefer an entry point that checks coverage before it gets
there: `Persistence.ImportSafeTensors` names every parameter first and refuses with an
`InvalidDataException` — "required model parameter '…' maps to no source tensor name
under the naming scheme — add a rule covering it" — which says *which* parameter is
uncovered.

A `null` is not ignored downstream either: export refuses a scheme that leaves any
weight unnamed, and import treats the parameter as one the scheme does not
cover — which then fails as a required parameter with no source tensor (see
[onnx-and-weights.md](onnx-and-weights.md#naming)).
