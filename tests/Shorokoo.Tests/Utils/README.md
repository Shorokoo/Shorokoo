# RetinaNet Tests

This directory contains tests for the Shorokoo RetinaNet implementation with ResNet50 backbone.

## Test Files

### RetinaNetDetectorTests.cs
Basic tests for RetinaNet model construction and execution:
- `TestRetinaNetTinyValidationWithComputation` - Tests RetinaNet with tiny backbone
- `TestRetinaNetTinyResNet18ValidationWithComputation` - Tests with ResNet18 backbone and tiny heads
- `TestFullRetinaNetResNet18ModelConstruction` - Tests full ResNet18 model construction
- `TestRetinaNetTinyResNet50ValidationWithComputation` - Tests with ResNet50 backbone and tiny heads
- `TestFullRetinaNetResNet50ModelConstruction` - Tests full ResNet50 model construction

**Domain**: Models  
**Tier**: Thorough

### RetinaNetNamingSchemes.cs
Parameter naming schemes for loading PyTorch weights into Shorokoo models:
- `CreateRetinaNetResNet50NoNormScheme()` - Maps 578 Shorokoo parameter IDs to 301 PyTorch parameters

This handles the complexity of:
- IfElse branches creating duplicate parameter paths
- Shared weights in classification and regression heads
- Different initialization schemes (InitSimple vs InitXavier)
- Complex nested module structures

### RetinaNetResNet50WeightLoadingTests.cs
Tests for loading PyTorch pretrained weights:
- `LoadRetinaNetResNet50PytorchSafeTensorsSimplePattern` - Loads torchvision weights and builds trained model
- `ValidateNamingSchemeCompleteness` - Validates that 100% of Shorokoo parameters can be mapped

**Domain**: Models  
**Tier**: Standard

### RetinaNetResNet50ExecutionTests.cs
Tests for validating inference outputs against PyTorch:
- `RunRetinaNetResNet50Model` - Compares Shorokoo outputs with PyTorch reference outputs
- `ValidateBackboneIntermediateLayers` - Validates backbone intermediate features (pending implementation)

**Domain**: Models  
**Tier**: Thorough

## Test Data Requirements

To run the weight loading and execution tests, you need:

1. **Model Weights** (130 MB):
   - Location: `tests/test-data/models/retinanet-resnet50/retinanet_resnet50_fpn.safetensors`
   - Source: PyTorch torchvision `retinanet_resnet50_fpn` with COCO_V1 weights

2. **Intermediate Layer Validation Data** (50 MB):
   - Location: `tests/test-data/integration/intermediate-layers/retinanet_resnet50_intermediates.safetensors`
   - Contains: input tensor and intermediate layer outputs from PyTorch for comparison

See `tests/test-data/models/retinanet-resnet50/README.md` for instructions on generating this data.

## Running Tests

```bash
# Run all RetinaNet tests
dotnet test --filter "FullyQualifiedName~RetinaNet"

# Run only weight loading tests (Standard tier)
dotnet test --filter "FullyQualifiedName~RetinaNetResNet50WeightLoadingTests"

# Run only execution validation tests (Thorough tier)
dotnet test --filter "FullyQualifiedName~RetinaNetResNet50ExecutionTests"

# Run only basic RetinaNet tests (Thorough tier)
dotnet test --filter "FullyQualifiedName~RetinaNetDetectorTests"
```

## Test Behavior Without Data

Tests gracefully skip if the required test data files are not present:
- Weight loading tests will print a message and skip
- Execution tests will check for both weights and intermediate data before running

This allows the tests to be part of the permanent test suite without requiring all developers to download large files.

## Architecture Notes

### RetinaNet ResNet50 NoNorm
The `RetinaNetResNet50NoNorm` variant matches torchvision's `retinanet_resnet50_fpn` exactly:
- **No normalization** in detection heads (unlike the standard RetinaNet which uses GroupNorm)
- ResNet50 backbone with FPN (Feature Pyramid Network)
- Detection heads with P3-P7 pyramid levels
- 9 anchors per location (3 scales × 3 aspect ratios)
- 91 classes (COCO: 80 classes + background)

### Parameter Mapping Challenge
Shorokoo generates 578 parameters while PyTorch has 301 because:
- **IfElse branches**: Both branches of conditional code paths generate parameters
- **Shared weights**: Detection head weights are shared across pyramid levels but appear 5 times in Shorokoo
- **Downsample parameters**: Exist for all blocks in templates, not just first blocks

The naming scheme correctly handles all these cases.

## Development Workflow

1. **Create/modify model** in `samples/RetinaNet/`
2. **Add basic tests** to validate model construction
3. **Generate test data** using Python scripts (if validating against PyTorch)
4. **Add weight loading tests** to validate parameter mapping
5. **Add execution tests** to validate numerical correctness

## References

- PyTorch reference: torchvision.models.detection.retinanet_resnet50_fpn
- Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
