#!/usr/bin/env python3
"""Generate the ResNet18 sample input tensor for release-test-plan check E-3.

Downloads the canonical PyTorch test image (a Samoyed, ImageNet class 258),
applies the timm `resnet18.a1_in1k` eval transform (bicubic resize with
crop_pct=0.95, center-crop 224, ImageNet mean/std), and writes a [1,3,224,224]
float32 tensor to `sample-input-dog.safetensors` next to this script.

The output is consumed by RealCheckpointPredictionTests; it is git-ignored
(like every *.safetensors here), so regenerate it locally before running the
`Purpose=Manual` prediction test.

Requires: `pip install numpy pillow safetensors` (no torch needed).
"""
import os
import urllib.request

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
IMAGE_PATH = os.path.join(HERE, "dog.jpg")
OUT_PATH = os.path.join(HERE, "sample-input-dog.safetensors")

IMG_SIZE, CROP_PCT = 224, 0.95
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"downloading {IMAGE_URL}")
        urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)

    im = Image.open(IMAGE_PATH).convert("RGB")
    scale = int(np.floor(IMG_SIZE / CROP_PCT))  # 235
    w, h = im.size
    if w <= h:
        nw, nh = scale, int(round(h * scale / w))
    else:
        nw, nh = int(round(w * scale / h)), scale
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - IMG_SIZE) // 2, (nh - IMG_SIZE) // 2
    im = im.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    arr = np.asarray(im, dtype=np.float32) / 255.0          # HWC in [0,1]
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))[None]                # NCHW [1,3,224,224]
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    save_file({"input": arr}, OUT_PATH)
    print(f"wrote {OUT_PATH}  shape={arr.shape}  dtype={arr.dtype}")


if __name__ == "__main__":
    main()
