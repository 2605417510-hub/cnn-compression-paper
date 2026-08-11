#!/usr/bin/env python3
# Post-training INT8 quantization (PTQ), FX graph mode, fbgemm backend.
#
# WHY PTQ AND NOT QAT
#   This study builds a unified metric framework, it does not try to maximise
#   quantized accuracy. PTQ needs no retraining and is enough to exercise the
#   framework. QAT would add a training variable to every cell of the matrix
#   and confound the comparison.
#
# WHY FX GRAPH MODE
#   Eager mode requires hand-inserting QuantStub/DeQuantStub and calling
#   fuse_modules manually. FX traces the graph and does both automatically.
#   If symbolic tracing fails on some model, fall back to eager mode.
#
# WHY fbgemm AND NOT qnnpack
#   qnnpack targets ARM and would fit the edge narrative better, but this
#   machine is x86; numbers from an ARM backend on x86 mean nothing. Measured
#   latency is therefore a proxy measurement, and the edge-platform claims
#   rest on the analytical roofline model instead.
#
# CALIBRATION SET - THE ONE RULE THAT CANNOT BE BROKEN
#   Activation ranges are unknown ahead of time, so a batch of real data must
#   flow through the model to observe them. Those images are drawn from the
#   TRAINING split with a fixed seed. Drawing them from the test split would
#   leak evaluation data into the model. That is a reject-on-sight problem.
#   No augmentation is applied: calibration must see deployment-time inputs.
#
# WHAT QUANTIZATION DOES AND DOES NOT CHANGE
#   Memory traffic drops to 1/4 (4 bytes per element becomes 1).
#   The multiply-accumulate COUNT is unchanged. Arithmetic intensity therefore
#   rises about 4x, pushing the model toward the compute-bound region.
#
# EXPECTED RESULT ON THIS MACHINE
#   Xeon E5-2603 v4 has AVX2 but no AVX-512 VNNI. fbgemm's fastest INT8 path
#   uses the VNNI fused dot-product instruction; without it the backend falls
#   back to a longer AVX2 sequence. Expect roughly 1.5-2x speedup, not 4x.
#   That shortfall is not a bug, it is the phenomenon this study characterises.
#
# Usage:
#   python quantize.py --model resnet18 \
#       --ckpt checkpoints/resnet18_c10_p00_fp32_s42_best.pth \
#       --name resnet18_c10_p00_int8_s42 --data-root /path/to/data

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.cifar_models import (
    build_model, MODEL_BUILDERS, NUM_CLASSES, NORM, INPUT_SIZE)
from profile_model import profile
from latency import measure


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_sets(dataset, data_root):
    """Calibration uses the TEST transform: no augmentation."""
    mean, std = NORM[dataset]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds_cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    train_set = ds_cls(data_root, train=True, download=True, transform=tf)
    test_set = ds_cls(data_root, train=False, download=True, transform=tf)
    return train_set, test_set


def make_calib_loader(train_set, n, seed, batch_size):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_set), generator=g)[:n].tolist()
    return DataLoader(Subset(train_set, idx), batch_size=batch_size,
                      shuffle=False, num_workers=2)


@torch.no_grad()
def evaluate(model, loader, limit=None):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        out = model(x)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
        if limit is not None and total >= limit:
            break
    return correct / total


def load_fp32(args):
    obj = torch.load(args.ckpt, map_location="cpu")
    # Pruned models are saved as whole modules: their architecture no longer
    # matches what build_model() produces.
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        return obj["model"]
    model = build_model(args.model, args.dataset)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state)
    return model


def file_mb(path):
    return os.path.getsize(path) / 1024.0 / 1024.0


def make_quantized_model(model_fp32, qconfig_mapping, example_inputs, calib_loader):
    """Prepare, calibrate, and convert a fresh copy of an FP32 model."""
    to_quant = copy.deepcopy(model_fp32).eval().cpu()
    prepared = quantize_fx.prepare_fx(to_quant, qconfig_mapping, example_inputs)

    t0 = time.time()
    with torch.no_grad():
        for x, _ in calib_loader:
            prepared(x)

    return quantize_fx.convert_fx(prepared), time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_BUILDERS), required=True)
    p.add_argument("--dataset", choices=list(NUM_CLASSES), default="cifar10")
    p.add_argument("--ckpt", required=True, help="FP32 checkpoint to quantize")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--name", default=None, help="output config id")
    p.add_argument("--calib-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--skip-rebuild-verify", action="store_true",
                   help="skip the state-dict reconstruction check")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    set_seed(args.seed)
    torch.backends.quantized.engine = "fbgemm"

    name = args.name or (
        os.path.splitext(os.path.basename(args.ckpt))[0]
        .replace("_fp32", "_int8").replace("_best", ""))

    for sub in ("checkpoints", "metrics"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    print("=" * 62)
    print("config   : %s" % name)
    print("source   : %s" % args.ckpt)
    print("backend  : fbgemm   engine: %s" % torch.backends.quantized.engine)
    print("=" * 62)

    # ------------------------------------------------------------ FP32 side
    model_fp32 = load_fp32(args).eval().cpu()
    train_set, test_set = build_sets(args.dataset, args.data_root)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    t0 = time.time()
    acc_fp32 = evaluate(model_fp32, test_loader)
    print("FP32 accuracy : %.2f%%   (%.1fs)" % (acc_fp32 * 100, time.time() - t0))

    fp32_size = file_mb(args.ckpt)

    # ------------------------------------------------------------ calibrate
    calib_loader = make_calib_loader(train_set, args.calib_size,
                                     args.seed, args.batch_size)
    print("calibration   : %d images from the TRAINING split, seed %d"
          % (args.calib_size, args.seed))

    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    example_inputs = (torch.randn(*INPUT_SIZE),)

    try:
        model_int8, calib_seconds = make_quantized_model(
            model_fp32, qconfig_mapping, example_inputs, calib_loader)
    except Exception as e:
        print("\nFX symbolic tracing FAILED:")
        print("  %s" % e)
        print("\nThe model has dynamic control flow that FX cannot trace.")
        print("Fall back to eager-mode PTQ: insert QuantStub/DeQuantStub and")
        print("call fuse_modules by hand. Roughly half a day of work.")
        raise SystemExit(1)

    print("calibrated    : %.1fs" % calib_seconds)

    # ------------------------------------------------------------ INT8 side
    t0 = time.time()
    acc_int8 = evaluate(model_int8, test_loader)
    print("INT8 accuracy : %.2f%%   (%.1fs)" % (acc_int8 * 100, time.time() - t0))

    # The FX GraphModule is not portable as a whole torch.save() object on
    # this PyTorch version. Measure it while it is still in memory, then save
    # only its state dict plus the deterministic reconstruction recipe.
    profile_res = profile(model_int8)

    ckpt_path = os.path.join(args.out_dir, "checkpoints",
                             name + "_state_dict.pth")
    artifact = {
        "format": "fx_quantized_state_dict_v1",
        "state_dict": model_int8.state_dict(),
        "model": args.model,
        "dataset": args.dataset,
        "source_ckpt": args.ckpt,
        "backend": "fbgemm",
        "calib_size": args.calib_size,
        "calib_split": "train",
        "seed": args.seed,
        "torch_version": torch.__version__,
    }
    torch.save(artifact, ckpt_path)
    int8_size = file_mb(ckpt_path)

    # ------------------------------------------------------------ rebuild verify
    # A direct torch.load(model_int8) fails for fused quantized FX modules.
    # Instead, rebuild the graph from the FP32 source and load its state dict.
    # This costs one extra calibration pass but proves that the saved artifact
    # is reproducible without depending on GraphModule pickling.
    rebuild_ok = None
    rebuild_error = None
    acc_rebuilt = None
    if not args.skip_rebuild_verify:
        try:
            saved = torch.load(ckpt_path, map_location="cpu")
            rebuilt, _ = make_quantized_model(
                model_fp32, qconfig_mapping, example_inputs, calib_loader)
            rebuilt.load_state_dict(saved["state_dict"], strict=True)
            acc_rebuilt = evaluate(rebuilt, test_loader, limit=2048)
            acc_int8_head = evaluate(model_int8, test_loader, limit=2048)
            rebuild_ok = abs(acc_rebuilt - acc_int8_head) < 1e-6
            if not rebuild_ok:
                rebuild_error = "reconstructed model disagrees with in-memory model"
        except Exception as e:
            rebuild_ok = False
            rebuild_error = "%s: %s" % (type(e).__name__, e)

    # Keep timing last: measure() pins PyTorch to one CPU thread.
    latency_res = measure(model_int8, INPUT_SIZE, args.warmup, args.repeats)
    latency_spread = (latency_res["latency_ms_std"] /
                      latency_res["latency_ms_median"] * 100.0)

    print("-" * 62)
    print("accuracy      : %.2f%% -> %.2f%%   (delta %+.2f)"
          % (acc_fp32 * 100, acc_int8 * 100, (acc_int8 - acc_fp32) * 100))
    print("file size     : %.2f MB -> %.2f MB   (%.2fx smaller)"
          % (fp32_size, int8_size,
             fp32_size / int8_size if int8_size else 0.0))
    if rebuild_ok is None:
        print("save/rebuild  : SKIPPED")
    else:
        print("save/rebuild  : %s" % ("OK" if rebuild_ok else "FAILED"))
    if rebuild_error:
        print("  %s" % rebuild_error)
    print("median latency: %.3f ms" % latency_res["latency_ms_median"])
    print("stability     : std/median = %.1f%%   %s" % (
        latency_spread, "OK" if latency_spread < 5.0 else "TOO NOISY"))
    print("=" * 62)

    if acc_fp32 - acc_int8 > 0.02:
        print("WARNING: accuracy dropped more than 2 points.")
        print("  Likely the depthwise layers. Options: keep the stem and the")
        print("  classifier in FP32 (mixed precision), and report it.")

    metrics = dict(profile_res)
    metrics.update(latency_res)
    metrics.update({
        "name": name,
        "model": args.model,
        "dataset": args.dataset,
        "source_ckpt": args.ckpt,
        "precision": "int8",
        "backend": "fbgemm",
        "mode": "fx_graph",
        "weight_observer": "per-channel MinMax",
        "activation_observer": "per-tensor histogram",
        "calib_size": args.calib_size,
        "calib_split": "train",
        "seed": args.seed,
        "acc_fp32": acc_fp32,
        "acc_int8": acc_int8,
        "acc_delta": acc_int8 - acc_fp32,
        "file_mb_fp32": fp32_size,
        "file_mb_int8": int8_size,
        "checkpoint_format": artifact["format"],
        "checkpoint_path": ckpt_path,
        "rebuild_verified": rebuild_ok,
        "rebuild_accuracy_head": acc_rebuilt,
        "rebuild_error": rebuild_error,
        "latency_std_over_median_pct": latency_spread,
        "torch_version": torch.__version__,
    })
    meta_path = os.path.join(args.out_dir, "metrics", name + ".json")
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("saved state dict : %s" % ckpt_path)
    print("metrics          : %s" % meta_path)


if __name__ == "__main__":
    main()
