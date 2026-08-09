#!/usr/bin/env python3
# Single source of truth for the CIFAR-adapted model definitions.
#
# WHY THIS FILE EXISTS
#   These builders were originally copy-pasted into train_baseline.py,
#   profile_model.py and latency.py. Three copies of the same code means a
#   change to one is a silent divergence from the others: the profiler would
#   then measure a network that is not the one being trained, no error would
#   be raised, and every derived number would be quietly wrong. One
#   definition, imported everywhere, makes that failure impossible.
#
# USAGE from a script in scripts/
#   import sys, os
#   sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
#   from models.cifar_models import build_model, NUM_CLASSES, NORM

import torch.nn as nn
from torchvision import models

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

# Normalization stats per dataset. These must be identical across training,
# evaluation and quantization calibration; any mismatch introduces a
# distribution shift and invalidates the quantized results.
NORM = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}

INPUT_SIZE = (1, 3, 32, 32)


def build_resnet18(num_classes):
    # torchvision ResNet targets ImageNet at 224x224:
    #   conv1 7x7 stride 2 and maxpool 3x3 stride 2 downsample by 4x up front.
    # On 32x32 that leaves 8x8 before the stages, and three further stage
    # strides collapse it to 1x1. Fix: 3x3 stride 1 stem, drop maxpool.
    # Total downsampling becomes 8x, final feature map 4x4.
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_mobilenetv2(num_classes):
    # Stock MobileNetV2 downsamples by 32x (stem plus four stride-2 stages),
    # collapsing 32x32 to 1x1. Remove two factors of two:
    #   1) stem features[0][0]: stride 2 -> 1
    #   2) first stride-2 inverted residual features[2] depthwise: stride 2 -> 1
    # Total downsampling becomes 8x, final feature map 4x4, matching ResNet-18.
    #
    # InvertedResidual.conv layout when expand_ratio != 1:
    #   conv[0] 1x1 pointwise expand   (Conv2dNormActivation)
    #   conv[1] 3x3 depthwise          (Conv2dNormActivation)  <- stride here
    #   conv[2] 1x1 pointwise project  (Conv2d)
    #   conv[3] BatchNorm2d
    m = models.mobilenet_v2(weights=None)

    m.features[0][0].stride = (1, 1)

    blk = m.features[2]
    assert blk.conv[1][0].stride == (2, 2), (
        "features[2] depthwise stride is not 2; torchvision layout differs, "
        "inspect the model before proceeding"
    )
    blk.conv[1][0].stride = (1, 1)
    blk.stride = 1

    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


MODEL_BUILDERS = {
    "resnet18": build_resnet18,
    "mobilenetv2": build_mobilenetv2,
}


def build_model(name, dataset="cifar10"):
    """Build a CIFAR-adapted model by name."""
    if name not in MODEL_BUILDERS:
        raise ValueError("unknown model %r, expected one of %s"
                         % (name, list(MODEL_BUILDERS)))
    if dataset not in NUM_CLASSES:
        raise ValueError("unknown dataset %r, expected one of %s"
                         % (dataset, list(NUM_CLASSES)))
    return MODEL_BUILDERS[name](NUM_CLASSES[dataset])


def classifier_module(model, name):
    """The layer that must never be pruned: its output width is the class count."""
    if name == "resnet18":
        return model.fc
    if name == "mobilenetv2":
        return model.classifier[1]
    raise ValueError("unknown model %r" % name)


# Reference values for the CIFAR-adapted networks, used as self-checks.
# ResNet-18 MACs were derived by hand:
#   stem 1,769,472 + layer1 150,994,944 + layer2/3/4 134,217,728 each
#   + fc 5,120 = 555,422,720
REFERENCE = {
    ("resnet18", "cifar10"): {"params": 11173962, "macs": 555422720},
    ("mobilenetv2", "cifar10"): {"params": 2236682, "macs": 87976448},
}


if __name__ == "__main__":
    import torch
    for name in MODEL_BUILDERS:
        m = build_model(name, "cifar10").eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            out = m(torch.randn(*INPUT_SIZE))
        ref = REFERENCE.get((name, "cifar10"), {})
        ok = "OK" if n == ref.get("params") else "MISMATCH"
        print("%-14s params %9d  out %s   %s"
              % (name, n, tuple(out.shape), ok))
