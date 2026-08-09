#!/usr/bin/env python3
# CPU single-thread inference latency measurement.
#
# WHY THE PROTOCOL IS WHAT IT IS
#   - CPU only: PyTorch post-training quantization runs on CPU only. If FP32
#     were timed on GPU and INT8 on CPU the two numbers would not be
#     comparable at all, and every speedup claim in the paper would collapse.
#   - single thread: thread scheduling is the largest source of variance in
#     CPU timing, and results change with thread count. Pin it to 1.
#   - batch size 1: edge inference processes one frame at a time.
#   - warmup: the first runs pay for lazy allocation, library load and cold
#     caches. Those samples are thrown away.
#   - median, not mean: latency is right-skewed with a heavy tail. A handful
#     of scheduling hiccups drag the mean up; the median does not move.
#
# The SAME code path times FP32 and INT8 models, so the comparison is clean.
#
# Usage:
#   python latency.py --model resnet18 --ckpt checkpoints/xxx_best.pth
#   python latency.py --ckpt checkpoints/pruned.pth            # whole-module ckpt
#   python latency.py --model resnet18 --random --repeats 300
#   python latency.py --model resnet18 --ckpt xxx.pth --merge metrics/xxx.json

import argparse
import json
import os
import platform
import statistics
import time

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def build_resnet18(num_classes):
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_mobilenetv2(num_classes):
    m = models.mobilenet_v2(weights=None)
    m.features[0][0].stride = (1, 1)
    blk = m.features[2]
    blk.conv[1][0].stride = (1, 1)
    blk.stride = 1
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


MODEL_BUILDERS = {
    "resnet18": build_resnet18,
    "mobilenetv2": build_mobilenetv2,
}


def load_model(args):
    num_classes = NUM_CLASSES[args.dataset]
    if args.random:
        return MODEL_BUILDERS[args.model](num_classes)

    obj = torch.load(args.ckpt, map_location="cpu")

    # A pruned or quantized model is saved as a whole module, because its
    # architecture no longer matches what the builder produces.
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        return obj["model"]

    if not args.model:
        raise SystemExit("this checkpoint holds only weights; pass --model")
    m = MODEL_BUILDERS[args.model](num_classes)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    m.load_state_dict(state)
    return m


def measure(model, input_size, warmup, repeats):
    torch.set_num_threads(1)          # THE critical line: single thread
    model.eval().cpu()
    x = torch.randn(*input_size)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            samples.append((time.perf_counter() - t0) * 1000.0)   # ms

    samples.sort()
    n = len(samples)
    return {
        "latency_ms_median": statistics.median(samples),
        "latency_ms_mean": statistics.fmean(samples),
        "latency_ms_std": statistics.pstdev(samples),
        "latency_ms_min": samples[0],
        "latency_ms_p95": samples[int(0.95 * (n - 1))],
        "latency_ms_max": samples[-1],
        "warmup": warmup,
        "repeats": repeats,
        "threads": torch.get_num_threads(),
        "batch_size": input_size[0],
        "cpu": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_BUILDERS), default=None)
    p.add_argument("--dataset", choices=list(NUM_CLASSES), default="cifar10")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--random", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--out", default=None, help="write a standalone JSON here")
    p.add_argument("--merge", default=None,
                   help="merge the timing into an existing profiler JSON")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    if not args.ckpt and not args.random:
        raise SystemExit("give --ckpt PATH or --random")

    model = load_model(args)
    res = measure(model, (1, 3, 32, 32), args.warmup, args.repeats)

    res["name"] = args.name or (
        os.path.splitext(os.path.basename(args.ckpt))[0] if args.ckpt
        else "%s_random" % args.model)

    spread = res["latency_ms_std"] / res["latency_ms_median"] * 100

    print("=" * 62)
    print("config  : %s" % res["name"])
    print("threads : %d   batch : %d   warmup : %d   repeats : %d"
          % (res["threads"], res["batch_size"], res["warmup"], res["repeats"]))
    print("-" * 62)
    print("median  : %8.3f ms      <- report this one" % res["latency_ms_median"])
    print("mean    : %8.3f ms" % res["latency_ms_mean"])
    print("std     : %8.3f ms  (%.1f%% of median)"
          % (res["latency_ms_std"], spread))
    print("min     : %8.3f ms" % res["latency_ms_min"])
    print("p95     : %8.3f ms" % res["latency_ms_p95"])
    print("max     : %8.3f ms" % res["latency_ms_max"])
    print("=" * 62)

    if spread < 5.0:
        print("stability: std/median = %.1f%%   OK" % spread)
    else:
        print("stability: std/median = %.1f%%   TOO NOISY" % spread)
        print("           something else is competing for the CPU.")
        print("           check with: nvidia-smi   and   top")
        print("           rerun when the machine is idle.")

    if args.merge:
        with open(args.merge) as f:
            data = json.load(f)
        data.update(res)
        with open(args.merge, "w") as f:
            json.dump(data, f, indent=2)
        print("merged  : %s" % args.merge)
    elif args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("written : %s" % args.out)


if __name__ == "__main__":
    main()
