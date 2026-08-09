#!/usr/bin/env python3
# Unified profiler: params / MACs / memory traffic in ONE traversal.
#
# Why not thop or ptflops:
#   1. MACs and memory bytes must come from the SAME forward pass and the SAME
#      shape derivation. If they disagree on any layer, arithmetic intensity is
#      wrong and the error is very hard to find.
#   2. thop does not recognise torch.ao.nn.quantized modules, and INT8 memory
#      traffic is core data for this study.
#   3. Pruned models have irregular channel counts and modified `groups`.
#
# Memory traffic model (upper bound, no on-chip reuse):
#   Mem_l = params_l * B + input_elems * B + output_elems * B
#   B = 4 for FP32, 1 for INT8
#   AI = MACs / Mem_total   [MACs/Byte]
#
# Usage:
#   python profile.py --model resnet18 --ckpt checkpoints/xxx_best.pth
#   python profile.py --model resnet18 --ckpt xxx.pth --out metrics/xxx.json
#   python profile.py --model resnet18 --random          # no checkpoint needed

import argparse
import json
import os

import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.cifar_models import (
    build_model, MODEL_BUILDERS, NUM_CLASSES, NORM, INPUT_SIZE)


try:
    import torch.ao.nn.quantized as nnq
    import torch.ao.nn.intrinsic.quantized as nniq
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False





# --------------------------------------------------------------- layer types
def conv_types():
    t = [nn.Conv2d]
    if HAS_QUANT:
        t += [nnq.Conv2d, nniq.ConvReLU2d]
    return tuple(t)


def linear_types():
    t = [nn.Linear]
    if HAS_QUANT:
        t += [nnq.Linear, nniq.LinearReLU]
    return tuple(t)


def is_leaf(module):
    return len(list(module.children())) == 0


def elem_bytes(tensor):
    # Quantized tensors report element_size() == 1 for qint8/quint8.
    try:
        return tensor.element_size()
    except Exception:
        return 4


def module_params(module):
    # Quantized modules hold packed weights, not plain Parameters, so
    # module.parameters() returns nothing. Recover shape from weight().
    n = sum(p.numel() for p in module.parameters(recurse=False))
    if n == 0 and hasattr(module, "weight"):
        try:
            w = module.weight() if callable(module.weight) else module.weight
            if w is not None:
                n = w.numel()
                b = getattr(module, "bias", None)
                b = b() if callable(b) else b
                if b is not None:
                    n += b.numel()
        except Exception:
            pass
    return n


def param_bytes(module, default_b):
    if hasattr(module, "weight"):
        try:
            w = module.weight() if callable(module.weight) else module.weight
            if w is not None:
                return module_params(module) * elem_bytes(w)
        except Exception:
            pass
    return module_params(module) * default_b


# --------------------------------------------------------------- profiler
class Profiler:
    def __init__(self, model):
        self.model = model
        self.records = []
        self.handles = []
        self.CONV = conv_types()
        self.LINEAR = linear_types()

    def _hook(self, name):
        def fn(module, inp, out):
            x = inp[0]
            if not torch.is_tensor(x) or not torch.is_tensor(out):
                return

            macs = 0
            if isinstance(module, self.CONV):
                # out: (N, C_out, H_out, W_out)
                c_out, h_out, w_out = out.shape[1], out.shape[2], out.shape[3]
                groups = getattr(module, "groups", 1)
                in_ch = module.in_channels
                kh, kw = module.kernel_size
                # CRITICAL: divide by groups. Depthwise conv has
                # groups == in_channels; omitting this inflates MACs enormously.
                macs = c_out * h_out * w_out * (in_ch // groups) * kh * kw
            elif isinstance(module, self.LINEAR):
                macs = module.in_features * module.out_features
            else:
                return

            b_in = elem_bytes(x)
            b_out = elem_bytes(out)
            p = module_params(module)
            pb = param_bytes(module, b_in)

            # per-sample basis: strip the batch dimension
            n = x.shape[0] if x.dim() > 0 else 1
            in_elems = x.numel() // n
            out_elems = out.numel() // n

            self.records.append({
                "name": name,
                "type": module.__class__.__name__,
                "params": p,
                "macs": int(macs),
                "weight_bytes": int(pb),
                "input_bytes": int(in_elems * b_in),
                "output_bytes": int(out_elems * b_out),
                "mem_bytes": int(pb + in_elems * b_in + out_elems * b_out),
                "out_shape": list(out.shape[1:]),
            })
        return fn

    def run(self, example_input):
        for name, m in self.model.named_modules():
            if is_leaf(m) and isinstance(m, self.CONV + self.LINEAR):
                self.handles.append(m.register_forward_hook(self._hook(name)))

        self.model.eval()
        with torch.no_grad():
            self.model(example_input)

        for h in self.handles:
            h.remove()
        self.handles = []
        return self.records


def profile(model, input_size=(1, 3, 32, 32)):
    x = torch.randn(*input_size)
    recs = Profiler(model).run(x)

    macs = sum(r["macs"] for r in recs)
    mem = sum(r["mem_bytes"] for r in recs)
    params_counted = sum(r["params"] for r in recs)
    params_total = sum(p.numel() for p in model.parameters())
    if params_total == 0:
        params_total = params_counted

    return {
        "params": int(params_total),
        "params_conv_linear": int(params_counted),
        "macs": int(macs),
        "mem_bytes": int(mem),
        "arithmetic_intensity": (macs / mem) if mem else 0.0,
        "num_layers": len(recs),
        "per_layer": recs,
    }


# --------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_BUILDERS), required=True)
    p.add_argument("--dataset", choices=list(NUM_CLASSES), default="cifar10")
    p.add_argument("--ckpt", default=None, help="path to a .pth checkpoint")
    p.add_argument("--random", action="store_true",
                   help="profile a freshly built model, no checkpoint")
    p.add_argument("--out", default=None, help="write JSON here")
    p.add_argument("--name", default=None, help="config id for the JSON")
    args = p.parse_args()

    if args.ckpt:
        obj = torch.load(args.ckpt, map_location="cpu")
        if isinstance(obj, dict) and "model" in obj:
            model = build_model(args.model, args.dataset)
            model.load_state_dict(obj["model"])
        elif isinstance(obj, nn.Module):
            # pruned models are saved as whole modules, since the
            # architecture no longer matches the builder
            model = obj
        else:
            model = build_model(args.model, args.dataset)
            model.load_state_dict(obj)
    elif args.random:
        model = build_model(args.model, args.dataset)
    else:
        raise SystemExit("give --ckpt PATH or --random")

    model.eval().cpu()
    res = profile(model)
    res["name"] = args.name or (
        os.path.splitext(os.path.basename(args.ckpt))[0] if args.ckpt
        else "%s_%s_random" % (args.model, args.dataset))
    res["model"] = args.model
    res["dataset"] = args.dataset

    print("=" * 62)
    print("config : %s" % res["name"])
    print("params : %d  (%.2f M)" % (res["params"], res["params"] / 1e6))
    print("MACs   : %d  (%.2f M)" % (res["macs"], res["macs"] / 1e6))
    print("memory : %d bytes  (%.2f MB)"
          % (res["mem_bytes"], res["mem_bytes"] / 1024 / 1024))
    print("AI     : %.3f MACs/Byte" % res["arithmetic_intensity"])
    print("layers : %d profiled" % res["num_layers"])
    print("=" * 62)

    # self-check: per-layer sums must equal the totals
    assert sum(r["macs"] for r in res["per_layer"]) == res["macs"]
    assert sum(r["mem_bytes"] for r in res["per_layer"]) == res["mem_bytes"]
    print("self-check: per-layer sums match totals   OK")

    if args.model == "resnet18" and args.dataset == "cifar10":
        expect = 555422720
        got = res["macs"]
        if got == expect:
            print("reference : MACs == %d   MATCH" % expect)
        else:
            print("reference : expected %d, got %d, diff %+d"
                  % (expect, got, got - expect))
            print("            (a mismatch is expected for pruned models)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("written : %s" % args.out)


if __name__ == "__main__":
    main()
