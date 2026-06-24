#!/usr/bin/env python3
"""Print the PyTorch reference logits for ResNet18 parity (release-test-plan E-3).

Runs the reference ResNet18 (standard torchvision architecture, loaded from the same
`resnet18.safetensors` checkpoint Shorokoo binds) on the exact same preprocessed input
tensor (`sample-input-dog.safetensors`) and prints the 1000 reference logits, plus a
ready-to-paste C# array literal. These values are baked into the test as
`RealCheckpointPredictionTests.ReferenceLogits` — there is no generated "golden" file;
this script exists only to (re)produce the baked numbers when the sample image,
preprocessing, or checkpoint change.

Feeding the identical input tensor to both frameworks isolates the comparison to
numerical kernel differences, so Shorokoo agrees with these to a few times 1e-4.

Requires: `pip install torch safetensors` (CPU torch is enough; torchvision/timm not needed).
Run `make-sample-input.py` first to produce the input tensor.
"""
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "resnet18.safetensors")
INPUT = os.path.join(HERE, "sample-input-dog.safetensors")


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make(64, 2)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def main():
    model = ResNet18()
    model.load_state_dict(load_file(CKPT), strict=True)
    model.eval()

    x = load_file(INPUT)["input"]
    with torch.no_grad():
        logits = model(x)[0]

    top1 = int(logits.argmax().item())
    print(f"top-1 class = {top1} (expected 258 = Samoyed), logit = {float(logits[top1]):.6f}")
    print("\nPaste into RealCheckpointPredictionTests.ReferenceLogits.cs:\n")
    vals = [f"{float(v):.6f}f" for v in logits]
    for i in range(0, len(vals), 8):
        print("        " + ", ".join(vals[i:i + 8]) + ",")


if __name__ == "__main__":
    main()
