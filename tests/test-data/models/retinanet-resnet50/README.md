# RetinaNet ResNet50 FPN Model Weights

This directory contains PyTorch pretrained weights for the RetinaNet ResNet50 FPN model from torchvision.

## Required Files

- `retinanet_resnet50_fpn.safetensors` - Model weights in safetensors format

## How to Generate Test Data

These weights are derived from torchvision's `retinanet_resnet50_fpn` (COCO_V1)
using PyTorch exporters: download and convert the pretrained weights to
safetensors, generate intermediate-layer outputs for validation, then place the
files at:

- `tests/test-data/models/retinanet-resnet50/retinanet_resnet50_fpn.safetensors`
- `tests/test-data/integration/intermediate-layers/retinanet_resnet50_intermediates.safetensors`

## Model Details

- **Architecture**: RetinaNet with ResNet50-FPN backbone
- **Source**: PyTorch torchvision `retinanet_resnet50_fpn`
- **Weights**: COCO_V1 (pretrained on COCO dataset)
- **Input Size**: 800×800 pixels (COCO standard)
- **Classes**: 91 (80 COCO classes + background)
- **Parameters**: 301 trainable parameters (265 backbone + 16 FPN + 20 heads)

## Intermediate Layers

The intermediate layer validation data includes:

- `input` - Original input tensor used for validation
- `backbone_conv1` - Output after first conv layer
- `backbone_bn1` - Output after first batch norm
- `backbone_layer1` - ResNet layer 1 output (C2)
- `backbone_layer2` - ResNet layer 2 output (C3)
- `backbone_layer3` - ResNet layer 3 output (C4)
- `backbone_layer4` - ResNet layer 4 output (C5)
- `classifications` - Final classification head outputs (concatenated P3-P7)
- `regressions` - Final regression head outputs (concatenated P3-P7)

## Notes

- The safetensors format is used for consistent cross-platform tensor storage
- The naming scheme in `RetinaNetNamingSchemes.cs` maps Shorokoo parameter IDs to PyTorch names
- Tests skip gracefully if these files are not present
- PyTorch conv weights are stored in [out_channels, in_channels, H, W] format by default

## Size

- Model weights: ~130 MB
- Intermediate layers: ~50 MB (depends on input size)
