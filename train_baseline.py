#!/usr/bin/env python3
# CIFAR baseline training: ResNet-18 / MobileNetV2, CIFAR-adapted.
# Output naming: {model}_{dataset}_p00_fp32_s{seed}

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.cifar_models import (
    build_model, MODEL_BUILDERS, NUM_CLASSES, NORM, INPUT_SIZE)




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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=workers, pin_memory=True)
    return train_loader, test_loader


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_BUILDERS), required=True)
    p.add_argument("--dataset", choices=list(NUM_CLASSES), default="cifar10")
    p.add_argument("--data-root", default="./data",
                   help="parent directory of cifar-10-batches-py")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_short = "c10" if args.dataset == "cifar10" else "c100"
    name = "%s_%s_p00_fp32_s%d" % (args.model, ds_short, args.seed)
    if args.scheduler == "none":
        name += "_fixedlr"
    if args.tag:
        name += "_" + args.tag

    for sub in ("checkpoints", "logs", "metrics"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    log_path = os.path.join(args.out_dir, "logs", name + "_curve.csv")
    metric_path = os.path.join(args.out_dir, "metrics", name + "_train.json")

    num_classes = NUM_CLASSES[args.dataset]
    model = build_model(args.model, args.dataset).to(device)

    model.eval()
    with torch.no_grad():
        n_params = sum(q.numel() for q in model.parameters())
        model(torch.randn(2, 3, 32, 32, device=device))
    print("[%s] params = %.2f M | device = %s" % (name, n_params / 1e6, device),
          flush=True)

    train_loader, test_loader = build_loaders(
        args.dataset, args.data_root, args.batch_size, args.workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc",
                                "test_loss", "test_acc", "sec"])

    best_acc, best_epoch = 0.0, -1
    t_start = time.time()
    tr_acc, te_acc = 0.0, 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0

        if te_acc > best_acc:
            best_acc, best_epoch = te_acc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "test_acc": te_acc, "args": vars(args)},
                       os.path.join(ckpt_dir, name + "_best.pth"))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, "%.6f" % cur_lr, "%.4f" % tr_loss,
                                    "%.4f" % tr_acc, "%.4f" % te_loss,
                                    "%.4f" % te_acc, "%.1f" % dt])

        print("ep %3d/%d | lr %.4f | train %.4f/%.2f%% | test %.4f/%.2f%% | "
              "best %.2f%%@%d | %.1fs"
              % (epoch, args.epochs, cur_lr, tr_loss, tr_acc * 100,
                 te_loss, te_acc * 100, best_acc * 100, best_epoch, dt),
              flush=True)

    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "test_acc": te_acc, "args": vars(args)},
               os.path.join(ckpt_dir, name + "_last.pth"))

    summary = {
        "name": name,
        "model": args.model,
        "dataset": args.dataset,
        "num_classes": num_classes,
        "params": n_params,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "final_test_acc": te_acc,
        "final_train_acc": tr_acc,
        "total_minutes": (time.time() - t_start) / 60.0,
        "hyperparams": vars(args),
        "normalize_mean": NORM[args.dataset][0],
        "normalize_std": NORM[args.dataset][1],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(metric_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE] %s | best %.2f%% @ep%d | %.1f min"
          % (name, best_acc * 100, best_epoch, summary["total_minutes"]),
          flush=True)


if __name__ == "__main__":
    main()
