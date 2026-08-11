#!/usr/bin/env python3
# Build a validated FP32/INT8 summary table from the eight metric pairs.
#
# Usage:
#   python summarize_metrics.py --metrics-dir metrics --out-dir results

import argparse
import csv
import glob
import json
import os


MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "mbv2": "MobileNetV2",
}

MODEL_ORDER = {"resnet18": 0, "mbv2": 1}

CSV_COLUMNS = [
    "config", "model", "pruning", "fp32_accuracy_pct", "int8_accuracy_pct",
    "accuracy_delta_pp", "params", "macs", "fp32_mem_bytes", "int8_mem_bytes",
    "memory_reduction_pct", "fp32_arithmetic_intensity",
    "int8_arithmetic_intensity", "ai_ratio", "fp32_latency_ms",
    "int8_latency_ms", "latency_speedup", "latency_reduction_pct",
    "fp32_file_mb", "int8_file_mb", "file_reduction_ratio",
    "fp32_latency_stability_pct", "int8_latency_stability_pct",
    "int8_rebuild_verified",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def require(data, key, path):
    if key not in data:
        raise ValueError("%s is missing required field %r" % (path, key))
    return data[key]


def parse_name(name):
    parts = name.split("_")
    try:
        dataset_index = parts.index("c10")
    except ValueError:
        raise ValueError("unexpected configuration name: %s" % name)

    model = "_".join(parts[:dataset_index])
    pruning = parts[dataset_index + 1]
    if model not in MODEL_LABELS or not pruning.startswith("p"):
        raise ValueError("unexpected configuration name: %s" % name)
    return model, pruning


def stability_pct(data, path):
    median = require(data, "latency_ms_median", path)
    std = require(data, "latency_ms_std", path)
    if median <= 0:
        raise ValueError("non-positive latency in %s" % path)
    return std / median * 100.0


def make_row(fp32, int8, fp32_path, int8_path):
    name = require(int8, "name", int8_path)
    model, pruning = parse_name(name)
    # The MobileNetV2 p00 FP32 JSON predates the short "mbv2" convention and
    # retains its training tag in the embedded display name. Pairing is based
    # on the deterministic matching filename created above, not that display
    # field. Validate the model and dataset fields instead.
    if require(fp32, "model", fp32_path) != require(int8, "model", int8_path):
        raise ValueError("model mismatch: %s vs %s" % (fp32_path, int8_path))
    if require(fp32, "dataset", fp32_path) != require(int8, "dataset", int8_path):
        raise ValueError("dataset mismatch: %s vs %s" % (fp32_path, int8_path))

    fp32_macs = require(fp32, "macs", fp32_path)
    int8_macs = require(int8, "macs", int8_path)
    verified_macs = require(int8, "macs_fp32_verified", int8_path)
    if fp32_macs != int8_macs or fp32_macs != verified_macs:
        raise ValueError("MAC mismatch for %s: FP32=%d INT8=%d verified=%d" % (
            name, fp32_macs, int8_macs, verified_macs))

    if require(int8, "rebuild_verified", int8_path) is not True:
        raise ValueError("INT8 rebuild validation failed for %s" % name)

    fp32_mem = require(fp32, "mem_bytes", fp32_path)
    int8_mem = require(int8, "mem_bytes", int8_path)
    fp32_ai = require(fp32, "arithmetic_intensity", fp32_path)
    int8_ai = require(int8, "arithmetic_intensity", int8_path)
    fp32_latency = require(fp32, "latency_ms_median", fp32_path)
    int8_latency = require(int8, "latency_ms_median", int8_path)
    fp32_size = require(int8, "file_mb_fp32", int8_path)
    int8_size = require(int8, "file_mb_int8", int8_path)

    return {
        "config": name.replace("_int8_s42", ""),
        "model": MODEL_LABELS[model],
        "_model_key": model,
        "pruning": pruning,
        "_pruning_value": int(pruning[1:]),
        "fp32_accuracy_pct": require(int8, "acc_fp32", int8_path) * 100.0,
        "int8_accuracy_pct": require(int8, "acc_int8", int8_path) * 100.0,
        "accuracy_delta_pp": require(int8, "acc_delta", int8_path) * 100.0,
        "params": require(fp32, "params", fp32_path),
        "macs": fp32_macs,
        "fp32_mem_bytes": fp32_mem,
        "int8_mem_bytes": int8_mem,
        "memory_reduction_pct": (1.0 - int8_mem / fp32_mem) * 100.0,
        "fp32_arithmetic_intensity": fp32_ai,
        "int8_arithmetic_intensity": int8_ai,
        "ai_ratio": int8_ai / fp32_ai,
        "fp32_latency_ms": fp32_latency,
        "int8_latency_ms": int8_latency,
        "latency_speedup": fp32_latency / int8_latency,
        "latency_reduction_pct": (1.0 - int8_latency / fp32_latency) * 100.0,
        "fp32_file_mb": fp32_size,
        "int8_file_mb": int8_size,
        "file_reduction_ratio": fp32_size / int8_size,
        "fp32_latency_stability_pct": stability_pct(fp32, fp32_path),
        "int8_latency_stability_pct": stability_pct(int8, int8_path),
        "int8_rebuild_verified": True,
    }


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    lines = [
        "# FP32 and INT8 experiment summary",
        "",
        "All rows passed the INT8 state-dict reconstruction check and the "
        "FP32/INT8 MAC-equivalence check.",
        "",
        "| Model | Pruning | FP32 Acc. | INT8 Acc. | Delta (pp) | Params (M) | "
        "MACs (M) | FP32 AI | INT8 AI | FP32 Lat. (ms) | INT8 Lat. (ms) | "
        "Speedup | INT8 Size (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {pruning} | {fp32_accuracy_pct:.2f} | "
            "{int8_accuracy_pct:.2f} | {accuracy_delta_pp:+.2f} | "
            "{params:.3f} | {macs:.3f} | {fp32_arithmetic_intensity:.3f} | "
            "{int8_arithmetic_intensity:.3f} | {fp32_latency_ms:.3f} | "
            "{int8_latency_ms:.3f} | {latency_speedup:.3f}x | "
            "{int8_file_mb:.2f} |".format(
                model=row["model"],
                pruning=row["pruning"],
                fp32_accuracy_pct=row["fp32_accuracy_pct"],
                int8_accuracy_pct=row["int8_accuracy_pct"],
                accuracy_delta_pp=row["accuracy_delta_pp"],
                params=row["params"] / 1e6,
                macs=row["macs"] / 1e6,
                fp32_arithmetic_intensity=row["fp32_arithmetic_intensity"],
                int8_arithmetic_intensity=row["int8_arithmetic_intensity"],
                fp32_latency_ms=row["fp32_latency_ms"],
                int8_latency_ms=row["int8_latency_ms"],
                latency_speedup=row["latency_speedup"],
                int8_file_mb=row["int8_file_mb"],
            ))
    lines.append("")
    with open(path, "w", newline="") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--expected-count", type=int, default=8)
    args = parser.parse_args()

    pattern = os.path.join(args.metrics_dir, "*_int8_s42.json")
    int8_paths = sorted(glob.glob(pattern))
    if len(int8_paths) != args.expected_count:
        raise SystemExit("expected %d INT8 metric files, found %d under %s" % (
            args.expected_count, len(int8_paths), args.metrics_dir))

    rows = []
    for int8_path in int8_paths:
        int8 = load_json(int8_path)
        fp32_name = require(int8, "name", int8_path).replace(
            "_int8_", "_fp32_") + ".json"
        fp32_path = os.path.join(args.metrics_dir, fp32_name)
        if not os.path.isfile(fp32_path):
            raise FileNotFoundError("missing matching FP32 metric: %s" % fp32_path)
        rows.append(make_row(load_json(fp32_path), int8, fp32_path, int8_path))

    rows.sort(key=lambda r: (MODEL_ORDER[r["_model_key"]], r["_pruning_value"]))
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "fp32_int8_summary.csv")
    markdown_path = os.path.join(args.out_dir, "fp32_int8_summary.md")
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)

    print("validated configurations: %d" % len(rows))
    print("csv     : %s" % csv_path)
    print("markdown: %s" % markdown_path)


if __name__ == "__main__":
    main()
