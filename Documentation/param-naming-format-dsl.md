# ModelId → Third-Party Name DSL

Related: [onnx-and-weights.md](onnx-and-weights.md) ·
[param-naming-pattern-dsl.md](param-naming-pattern-dsl.md)

Reference for the format strings accepted by `ModelIdFormat` /
`ModelIdNamingScheme` — one of the two ways to map parameter names when
[binding third-party weights](onnx-and-weights.md#bind-loaded-weights-into-a-model-for-inference)
(e.g. PyTorch/timm checkpoints) into a model with
`ToConcreteModel(weights, namingScheme)`. The alternative, matching on
Shorokoo ID strings instead of ModelIds, is the
[pattern DSL](param-naming-pattern-dsl.md).

A simple domain-specific language for converting Shorokoo ModelIds to third-party framework parameter names.

## 1. Overview

ModelIds are integer arrays that uniquely identify parameters in a Shorokoo model. This DSL provides format strings that reference array positions to construct parameter names.

```csharp
// ModelId: [1, 2, 1, 0, 1, 1, 1]
//           ↑  ↑  ↑  ↑  ↑  ↑  ↑
//          [0][1][2][3][4][5][6]

var scheme = new ModelIdFormat(
    format: "layer{1}.{3}.conv{5}.{6|weight,bias}"
);

// Result: "layer2.0.conv1.bias"
//   {6|…} reads the value at index 6, which is 1, and inline map entries are
//   0-based (§3.2) — so entry 1, "bias".
```

## 2. Escape Sequences

| Syntax | Produces | Use Case |
|--------|----------|----------|
| `\o` | `{` | Literal opening brace |
| `\c` | `}` | Literal closing brace |
| `\s` | `\` | Literal backslash |

Note: a literal `}` in text outside a placeholder is passed through unchanged, so
writing it as `\c` is optional. `ModelIdFormat.EscapeString` — which Shorokoo uses when
it generates a scheme for you — nevertheless emits `\c` for *every* `}`, so a generated
format string is full of them; `ModelIdFormat.UnescapeString` decodes all three
sequences. (The [pattern DSL](param-naming-pattern-dsl.md) has its own, smaller escape
set; the two are not interchangeable.)

## 3. Format String Syntax

### 3.1 Basic Placeholders

| Syntax | Description | Example |
|--------|-------------|---------|
| `{N}` | Value at index N | `{1}` → "2" |
| `{N + M}` | Add offset M | `{1 + 1}` → "3" |
| `{N - M}` | Subtract offset M | `{1 - 1}` → "1" |
| `text` | Literal text | `layer` → "layer" |

### 3.2 Inline Maps

Map index values to strings using comma-separated values:

```csharp
format: "{6|weight,bias}"
// Value at index 6 is 0 → "weight"
// Value at index 6 is 1 → "bias"
```

Entries are 0-based, and a value outside the list throws `IndexOutOfRangeException`.

### 3.3 Named Maps

Reference reusable maps for complex mappings:

```csharp
format: "{5|moduleMap}.{6|paramMap}",
maps: new()
{
    ["moduleMap"] = new() { [0] = "conv1", [1] = "bn1", [2] = "conv2" },
    ["paramMap"]  = new() { [0] = "weight", [1] = "bias" }
}

// ModelId [1, 2, 1, 0, 1, 1, 1] → "bn1.bias"
```

A named map is keyed by value, not by position, so its keys need not be contiguous —
but a value with no key throws `KeyNotFoundException`. The name must be declared in
`maps`; anything else after the `|` is read as an inline map instead.

### 3.4 Range Matching with Maps

Map numeric ranges to different outputs:

**Syntax:** `{N|ranges|outputs}`

`ranges` and `outputs` are both comma-separated lists, paired position by position:
the first range that matches the value at index N selects the output beside it. The
comma is the separator *between* entries — it is never an "or" inside one range. The
two lists must be the same length (`FormatException` otherwise), and a value matching
no range throws `KeyNotFoundException`.

| Range Syntax | Matches |
|--------------|---------|
| `1` | Only 1 |
| `2:5` | 2, 3, 4, 5 (start to end, inclusive) |
| `3:` | 3 and above |
| `3::2` | 3, 5, 7, ... (start at 3, step 2, no upper bound) |
| `2::2` | 2, 4, 6, ... (even numbers ≥ 2) |
| `2:8:2` | 2, 4, 6, 8 (start, end, step — end is inclusive) |
| `:6:2` | 0, 2, 4, 6 (an omitted start means 0) |

To match 1 or 3, give each its own entry and repeat the output: `{5|1,3|conv,conv}`.

**Example:**

```csharp
format: "{5|1,3::2,2::2|conv,bn,layer}"

// value at index 5 = 1 → "conv"
// value at index 5 = 2 → "layer"  (matches 2::2)
// value at index 5 = 3 → "bn"     (matches 3::2)
// value at index 5 = 4 → "layer"  (matches 2::2)
// value at index 5 = 5 → "bn"     (matches 3::2)
```

### 3.5 Recursive Format Strings

Embed placeholders within map outputs:

```csharp
format: "{5|1,3::2,2::2|conv,bn{5},new_{5|2::4,4::4|layer,fc}}"

// value at index 5 = 1 → "conv"
// value at index 5 = 2 → "new_layer"   (2::2, then 2::4)
// value at index 5 = 3 → "bn3"         (3::2)
// value at index 5 = 4 → "new_fc"      (2::2, then 4::4)
// value at index 5 = 5 → "bn5"
// value at index 5 = 6 → "new_layer"   (2::4)
// value at index 5 = 7 → "bn7"
// value at index 5 = 8 → "new_fc"      (4::4)
```

An index expression is parsed as a number, so a placeholder must name a position —
`{5|…}`, never `{idx|…}`.

## 4. Match Patterns

Filter which ModelIds a scheme applies to:

| Pattern | Matches | Description |
|---------|---------|-------------|
| `[1, 2, 1, *, *, *, *]` | Fixed positions + wildcards | Exact match with any values |
| `[1, 3\|4\|5, *, *, *, *, *]` | OR for positions | Layer 2, 3, or 4 |
| `[1, *, 1, *, *, 7\|8, *]` | Specific positions | Downsample modules |
| `-1` (in a position) | Any value | Same as `*` |
| `*` | All ModelIds | Universal fallback |

A bracketed pattern only matches a ModelId of exactly the same length; a pattern with
too few or too many positions never matches. Omitting `match` entirely also matches
every ModelId.

## 5. Complete ResNet50 Example

### 5.1 Scheme Definition

```csharp
ModelIdFormat[] formats =
[
    // ════════════════════════════════════════════════════════════════
    // STEM: [1, 1, modType, paramIdx]
    // ════════════════════════════════════════════════════════════════
    new ModelIdFormat(
        match: "[1, 1, *, *]",
        format: "{2|conv1,bn1}.{3|weight,running_mean,running_var,weight,bias}"
    ),

    // ════════════════════════════════════════════════════════════════
    // LAYER 1: [1, 2, 1, loop, block, mod, param]
    // ════════════════════════════════════════════════════════════════
    new ModelIdFormat(
        match: "[1, 2, 1, *, *, *, *]",
        format: "layer1.{3}.{5|conv,bn,conv,bn,conv,bn,downsample.0,downsample.1}{5|1,1,2,2,3,3,.,.}.{6|weight,running_mean,running_var,weight,bias}"
    ),

    // ════════════════════════════════════════════════════════════════
    // LAYERS 2-4: [1, layer, 1, loop, block, mod, param]
    // layer index: 3→layer2, 4→layer3, 5→layer4
    // ════════════════════════════════════════════════════════════════
    new ModelIdFormat(
        match: "[1, 3|4|5, 1, *, *, *, *]",
        format: "layer{1 - 1}.{3}.{5|conv,bn,conv,bn,conv,bn,downsample.0,downsample.1}{5|1,1,2,2,3,3,.,.}.{6|weight,running_mean,running_var,weight,bias}"
    ),

    // ════════════════════════════════════════════════════════════════
    // FC: [1, 6, 1, param]
    // ════════════════════════════════════════════════════════════════
    new ModelIdFormat(
        match: "[1, 6, 1, *]",
        format: "fc.{3|weight,bias}"
    ),
];

var scheme = new ModelIdNamingScheme(formats, ModuleParamSetNamingScheme.PyTorchFrameworkId);

// [1, 1, 0, 0]          → "conv1.weight"
// [1, 2, 1, 0, 0, 0, 0] → "layer1.0.conv1.weight"
// [1, 6, 1, 1]          → "fc.bias"
```

### 5.2 Step-by-Step Example

**ModelId:** `[1, 3, 1, 2, 2, 3, 1]`

**Format:** `"layer{1 - 1}.{3}.{5|conv,bn,...}{5|1,1,2,2,...}.{6|weight,...}"`

| Step | Token | Evaluation | Output |
|------|-------|------------|--------|
| 1 | `layer` | Literal | "layer" |
| 2 | `{1 - 1}` | 3 - 1 = 2 | "2" |
| 3 | `.` | Literal | "." |
| 4 | `{3}` | Index 3 = 2 | "2" |
| 5 | `.` | Literal | "." |
| 6 | `{5\|conv,bn,conv,bn,...}` | Index 5 = 3 → "bn" | "bn" |
| 7 | `{5\|1,1,2,2,3,3,...}` | Index 5 = 3 → "2" | "2" |
| 8 | `.` | Literal | "." |
| 9 | `{6\|weight,running_mean,...}` | Index 6 = 1 → entry 1 | "running_mean" |

**Result:** `"layer2.2.bn2.running_mean"`

### 5.3 Named Maps Alternative

```csharp
ModelIdFormat[] formats =
[
    new ModelIdFormat(
        match: "[1, 3|4|5, 1, *, *, *, *]",
        format: "layer{1 - 1}.{3}.{5|moduleMap}.{6|paramMap}",
        maps: new()
        {
            ["moduleMap"] = new()
            {
                [0] = "conv1", [1] = "bn1",
                [2] = "conv2", [3] = "bn2",
                [4] = "conv3", [5] = "bn3",
                [6] = "downsample.0", [7] = "downsample.1"
            },
            ["paramMap"] = new()
            {
                [0] = "weight",
                [1] = "running_mean",
                [2] = "running_var",
                [3] = "weight",
                [4] = "bias"
            }
        }
    ),
];

var namedMapScheme = new ModelIdNamingScheme(formats, ModuleParamSetNamingScheme.PyTorchFrameworkId);

// [1, 3, 1, 2, 2, 3, 1] → "layer2.2.bn2.running_mean", as in §5.2
```

## 6. Best Practices

### Order Schemes Specific to General

```csharp
ModelIdFormat[] formats =
[
    new ModelIdFormat(match: "[1, 1, *, *]", ...),           // Stem (most specific)
    new ModelIdFormat(match: "[1, 2, 1, *, *, *, *]", ...),  // Layer1
    new ModelIdFormat(match: "[1, 3|4|5, 1, *, *, *, *]", ...),  // Layers 2-4
    new ModelIdFormat(match: "[1, 6, 1, *]", ...),           // FC
    new ModelIdFormat(match: "*", ...)                       // Fallback
];
```

The first matching format wins, so a `"*"` fallback must come last.

### Use Offsets for Index Shifts

```csharp
// Shorokoo layer index 3 → PyTorch layer2
format: "layer{1 - 1}.{3}.conv{5}.weight"
```

## 7. API Reference

### ModelIdFormat

```csharp
public class ModelIdFormat
{
    public ModelIdFormat(
        string format,
        string? match = null,
        Dictionary<string, Dictionary<int, string>>? maps = null
    );

    public string Format { get; }
    public string? Match { get; }
    public ImmutableDictionary<string, ImmutableDictionary<int, string>> Maps { get; }

    public bool Matches(ModelId modelId);
    public string ToName(ModelId modelId);

    public static string EscapeString(string input);    // { → \o, } → \c, \ → \s
    public static string UnescapeString(string input);
}
```

### ModelIdNamingScheme

```csharp
public class ModelIdNamingScheme : ModuleParamSetNamingScheme
{
    // frameworkId is required: it records which framework's names this scheme speaks,
    // e.g. ModuleParamSetNamingScheme.PyTorchFrameworkId.
    public ModelIdNamingScheme(IEnumerable<ModelIdFormat> patterns, string frameworkId);

    public ImmutableArray<ModelIdFormat> Patterns { get; }
    public string FrameworkId { get; }                  // from ModuleParamSetNamingScheme

    public override string ToName(ModelId modelId);
    public override string ToName(ConcreteModelParamInfo shorokooParam);
    public override ModelId? ToModelId(string paramName, ImmutableArray<ModelId> candidates);
}
```

`ToModelId` is the reverse direction used when binding weights: it names every
candidate ModelId once and looks the third-party name up in that table, returning null
when nothing matches. The inherited `ToName(string shorokooId)` overload throws
`NotSupportedException` — a scheme keyed on ModelIds cannot translate a canonical
Shorokoo id string; use a `SimplePatternNamingScheme`
([pattern DSL](param-naming-pattern-dsl.md)) where that direction is needed, as weight
export does.

## 8. Error Handling

The DSL throws standard BCL exception types; there are no DSL-specific exception
classes.

| Failure | Exception |
|---------|-----------|
| No format in the scheme matches the ModelId | `InvalidOperationException` |
| Placeholder references a position the ModelId does not have | `IndexOutOfRangeException` |
| Value outside an inline map's entries | `IndexOutOfRangeException` |
| Value has no key in a named map | `KeyNotFoundException` |
| Value matches none of a range map's ranges | `KeyNotFoundException` |
| Range count differs from output count | `FormatException` |
| Unmatched `{` in the format string | `FormatException` |

```csharp
// No matching format
try { var name = scheme.ToName(unknownId); }
catch (InvalidOperationException ex) { /* "No matching pattern for ModelId [...]" */ }

// Placeholder index past the end of the ModelId
try { var name = new ModelIdFormat(format: "conv{5}").ToName(new ModelId(1, 2, 3)); }
catch (IndexOutOfRangeException ex) { /* Format references invalid index */ }

// Named map (§5.3) has no key 99 — an inline map would throw IndexOutOfRangeException
try { var name = namedMapScheme.ToName(new ModelId(1, 3, 1, 2, 2, 99, 1)); }
catch (KeyNotFoundException ex) { /* Key 99 not in map 'moduleMap' */ }
```
