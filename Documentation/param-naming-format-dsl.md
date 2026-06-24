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

// Result: "layer2.0.conv1.weight"
```

## 2. Escape Sequences

| Syntax | Produces | Use Case |
|--------|----------|----------|
| `\o` | `{` | Literal opening brace |
| `\s` | `\` | Literal backslash |

Note: Closing brace `}` does not require escaping.

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
// Index 0 → "weight"
// Index 1 → "bias"
```

### 3.3 Named Maps

Reference reusable maps for complex mappings:

```csharp
format: "{5|moduleMap}.{6|paramMap}",
maps: new()
{
    ["moduleMap"] = new() { [0] = "conv1", [1] = "bn1", [2] = "conv2" },
    ["paramMap"]  = new() { [0] = "weight", [1] = "bias" }
}
```

### 3.4 Range Matching with Maps

Map numeric ranges to different outputs:

**Syntax:** `{N|ranges|outputs}`

| Range Syntax | Matches |
|--------------|---------|
| `1` | Only 1 |
| `1,3` | 1 or 3 |
| `3::2` | 3, 5, 7, ... (start at 3, step 2) |
| `2::2` | 2, 4, 6, ... (even numbers ≥ 2) |

**Example:**

```csharp
format: "{idx|1,3::2,2::2|conv,bn,layer}"

// 1      → "conv"
// 2      → "layer"  (matches 2::2)
// 3      → "bn"     (matches 3::2)
// 4      → "layer"  (matches 2::2)
// 5      → "bn"     (matches 3::2)
```

### 3.5 Recursive Format Strings

Embed placeholders within map outputs:

```csharp
format: "{idx|1,3::2,2::2|conv,bn{idx},new_{idx|2::4,4::4|layer,fc}}"

// 1 → "conv"
// 2 → "new_layer"   (2%4≠0)
// 3 → "bn3"
// 4 → "new_fc"      (4%4=0)
// 5 → "bn5"
// 6 → "new_layer"   (6%4≠0)
// 7 → "bn7"
// 8 → "new_fc"      (8%4=0)
```

## 4. Match Patterns

Filter which ModelIds a scheme applies to:

| Pattern | Matches | Description |
|---------|---------|-------------|
| `[1, 2, 1, *, *, *, *]` | Fixed positions + wildcards | Exact match with any values |
| `[1, 3\|4\|5, *, *, *, *, *]` | OR for positions | Layer 2, 3, or 4 |
| `[1, *, 1, *, *, 7\|8, *]` | Specific positions | Downsample modules |
| `*` | All ModelIds | Universal fallback |

## 5. Complete ResNet50 Example

### 5.1 Scheme Definition

```csharp
var schemes = new ModelIdNamingScheme(new[]
{
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
    )
});
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
| 9 | `{6\|weight,...}` | Index 6 = 1 → "weight" | "weight" |

**Result:** `"layer2.2.bn2.weight"`

### 5.3 Named Maps Alternative

```csharp
var schemes = new ModelIdNamingScheme(new[]
{
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
    )
});
```

## 6. Best Practices

### Order Schemes Specific to General

```csharp
var schemes = new[]
{
    new ModelIdFormat(match: "[1, 1, *, *]", ...),           // Stem (most specific)
    new ModelIdFormat(match: "[1, 2, 1, *, *, *, *]", ...),  // Layer1
    new ModelIdFormat(match: "[1, 3|4|5, 1, *, *, *, *]", ...),  // Layers 2-4
    new ModelIdFormat(match: "[1, 6, 1, *]", ...),           // FC
    new ModelIdFormat(match: "*", ...)                       // Fallback
};
```

### Use Offsets for Index Shifts

```csharp
// Shorokoo layer index 3 → PyTorch layer2
format: "layer{1 - 1}.{3}.conv{5}.weight"
```

## 7. API Reference

### ModelIdFormat Constructor

```csharp
public ModelIdFormat(
    string format,
    string? match = null,
    Dictionary<string, Dictionary<int, string>>? maps = null
)
```

### ModelIdNamingScheme

```csharp
public class ModelIdNamingScheme
{
    public ModelIdNamingScheme(IEnumerable<ModelIdFormat> formats);
    public string ToName(ModelId id);
    public string ToName(int[] id);
    public bool CanConvert(ModelId id);
    public ModelIdFormat? GetMatchingFormat(ModelId id);
}
```

## 8. Error Handling

```csharp
// No matching format
try { var name = scheme.ToName(unknownId); }
catch (NoMatchingFormatException ex) { /* Handle */ }

// Index out of range
try { var name = scheme.ToName([1, 2, 3]); }
catch (IndexOutOfRangeException ex) { /* Format references invalid index */ }

// Map key not found
try { var name = scheme.ToName([1, 2, 1, 0, 1, 99, 1]); }
catch (MapKeyNotFoundException ex) { /* Key 99 not in map */ }
```
