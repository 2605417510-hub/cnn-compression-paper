#!/usr/bin/env python3
# Structured channel pruning with iterative schedule and fine-tuning.
#
# WHY STRUCTURED, NOT UNSTRUCTURED
#   Unstructured pruning zeroes individual weights and leaves the tensor
#   shape unchanged. Sparse matmul on general-purpose CPUs needs dedicated
#   kernels and indexing overhead, and below ~90% sparsity is often slower
#   than the dense computation it replaces. Parameter count drops, latency
#   does not. This study converts compression into end-to-end time, so a
#   method whose latency does not respond to compression makes the whole
#   decomposition meaningless.
#
# WHY GROUP-LEVEL IMPORTANCE
#   Residual connections force channel coupling: the two branches feeding an
#   addition must carry identical channel counts, so a channel removed from
#   one must be removed from the other. torch-pruning's DepGraph finds all
#   coupled groups automatically, and importance must therefore be scored per
#   group, not per layer in isolation.
#
# WHY ITERATIVE
#   Removing 70% of channels in one pass frequently damages the network
#   beyond recovery. Five smaller steps, each followed by a short fine-tune,
#   cost about the same in total epochs but retain far more accuracy.
#
# WHY round_to=8
#   Vector units operate on fixed-width lanes. A layer left with 37 channels
#   executes as though it had 40. This study is about the divergence between
#   theoretical and realized cost, so a self-inflicted divergence would
#   confound the analysis. Cost: the achieved ratio deviates from the
#   configured one, so BOTH are recorded.
#
# CHECKPOINT FORMAT
#   Pruned models are saved as WHOLE MODULES via torch.save(model), not as
#   state dicts. The architecture no longer matches what build_model()
#   produces, so a state dict could not be loaded back.
#
# Usage:
#   python prune.py --model resnet18 --ckpt checkpoints/resnet18_c10_p00_fp32_s42_best.pth --ratio 0.3
#   python prune.py --model mobilenetv2 --ckpt ... --ratio 0.5 --data-root /path/to/data

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.cifar_models import (
    build_model, MODEL_BUILDERS, NUM_CLASSES, NORM, INPUT_SIZE,
    classifier_module)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(dataset, data_root, batch_size, workers):
    mean, std = NORM[dataset]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds_cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    train_set = ds_cls(data_root, train=True, download=True, transform=train_tf)
    test_set = ds_cls(data_root, train=False, download=True, transform=test_tf)
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True,
                       num_workers=workers, pin_memory=True),
            DataLoader(test_set, batch_size=256, shuffle=False,
                       num_workers=workers, pin_memory=True))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def finetune(model, train_loader, test_loader, device, epochs, lr,
             cosine, log_rows, phase, best):
    if epochs <= 0:
        return best
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
             if cosine else None)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        if sched is not None:
            sched.step()
        dt = time.time() - t0
        if te_acc > best["acc"]:
            best["acc"] = te_acc
            best["state"] = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
        log_rows.append([phase, ep, "%.4f" % tr_loss, "%.4f" % tr_acc,
                         "%.4f" % te_loss, "%.4f" % te_acc, "%.1f" % dt])
        print("  [%s] ep %2d/%d | train %.4f/%.2f%% | test %.4f/%.2f%% | "
              "best %.2f%% | %.1fs"
              % (phase, ep, epochs, tr_loss, tr_acc * 100,
                 te_loss, te_acc * 100, best["acc"] * 100, dt), flush=True)
    return best


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_BUILDERS), required=True)
    p.add_argument("--dataset", choices=list(NUM_CLASSES), default="cifar10")
    p.add_argument("--ckpt", required=True, help="baseline checkpoint")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--ratio", type=float, required=True,
                   help="channel pruning ratio, e.g. 0.3")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--step-epochs", type=int, default=5)
    p.add_argument("--final-epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--round-to", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_short = "c10" if args.dataset == "cifar10" else "c100"
    short = {"resnet18": "resnet18", "mobilenetv2": "mbv2"}[args.model]
    name = "%s_%s_p%02d_fp32_s%d" % (short, ds_short,
                                     int(round(args.ratio * 100)), args.seed)

    for sub in ("checkpoints", "logs", "metrics"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    # ---------------------------------------------------------- load baseline
    obj = torch.load(args.ckpt, map_location="cpu")
    model = build_model(args.model, args.dataset)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state)
    model.to(device)

    params_before = count_params(model)
    print("=" * 62)
    print("config      : %s" % name)
    print("baseline    : %s" % args.ckpt)
    print("params      : %d  (%.2f M)" % (params_before, params_before / 1e6))
    print("target ratio: %.0f%%   steps: %d   round_to: %d"
          % (args.ratio * 100, args.steps, args.round_to))
    print("=" * 62)

    train_loader, test_loader = build_loaders(
        args.dataset, args.data_root, args.batch_size, args.workers)
    criterion = nn.CrossEntropyLoss()

    _, acc_before = evaluate(model, test_loader, criterion, device)
    print("baseline accuracy re-checked: %.2f%%" % (acc_before * 100))
    print("-" * 62)

    # ---------------------------------------------------------- pruner
    example_inputs = torch.randn(*INPUT_SIZE).to(device)

    # The classifier must never be pruned: its output width is the class count.
    ignored = [classifier_module(model, args.model)]

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=args.ratio,
        iterative_steps=args.steps,
        round_to=args.round_to,
        ignored_layers=ignored,
    )

    log_rows = []
    best = {"acc": 0.0, "state": None}

    for step in range(1, args.steps + 1):
        pruner.step()
        now = count_params(model)
        print("step %d/%d  params %d (%.2f M, -%.1f%%)"
              % (step, args.steps, now, now / 1e6,
                 100.0 * (1 - now / params_before)), flush=True)
        best = finetune(model, train_loader, test_loader, device,
                        args.step_epochs, args.lr, False,
                        log_rows, "step%d" % step, best)

    print("-" * 62)
    print("final fine-tune, %d epochs, cosine" % args.final_epochs)
    best = finetune(model, train_loader, test_loader, device,
                    args.final_epochs, args.lr, True,
                    log_rows, "final", best)

    # restore the best weights seen
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # ---------------------------------------------------------- report
    params_after = count_params(model)
    actual = 1.0 - params_after / params_before

    model.eval().cpu()
    with torch.no_grad():
        out = model(torch.randn(*INPUT_SIZE))
    assert out.shape[1] == NUM_CLASSES[args.dataset], \
        "classifier width changed, it should have been in ignored_layers"

    print("=" * 62)
    print("config          : %s" % name)
    print("params          : %d -> %d" % (params_before, params_after))
    print("set ratio       : %.3f   (channels)" % args.ratio)
    print("param reduction : %.3f" % actual)
    print("accuracy        : %.2f%% -> %.2f%%  (delta %+.2f)"
          % (acc_before * 100, best["acc"] * 100,
             (best["acc"] - acc_before) * 100))
    print("forward check   : output %s   OK" % (tuple(out.shape),))
    print("=" * 62)

    # WHOLE MODULE, not a state dict: the architecture no longer matches
    # what build_model() would produce.
    ckpt_path = os.path.join(args.out_dir, "checkpoints", name + "_best.pth")
    torch.save(model, ckpt_path)
    print("saved  : %s" % ckpt_path)

    log_path = os.path.join(args.out_dir, "logs", name + "_curve.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "epoch", "train_loss", "train_acc",
                    "test_loss", "test_acc", "sec"])
        w.writerows(log_rows)
    print("logged : %s" % log_path)

    meta = {
        "name": name,
        "model": args.model,
        "dataset": args.dataset,
        "baseline_ckpt": args.ckpt,
        "baseline_acc": acc_before,
        "top1": best["acc"],
        "params_before": params_before,
        "params_after": params_after,
        "set_prune_ratio": args.ratio,
        "actual_param_reduction": actual,
        "steps": args.steps,
        "round_to": args.round_to,
        "step_epochs": args.step_epochs,
        "final_epochs": args.final_epochs,
        "finetune_lr": args.lr,
        "seed": args.seed,
        "importance": "GroupMagnitudeImportance(p=2)",
        "torch_version": torch.__version__,
        "torch_pruning_version": getattr(tp, "__version__", "unknown"),
    }
    meta_path = os.path.join(args.out_dir, "metrics", name + "_prune.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("meta   : %s" % meta_path)


if __name__ == "__main__":
    main()
