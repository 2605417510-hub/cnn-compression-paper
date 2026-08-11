#!/usr/bin/env python3
# Produce a layer-level roofline time-constant decomposition.
#
# Usage:
#   python analyze_tau_layers.py --metrics-dir metrics --out-dir results \
#       --peak-gmac-s 24.36 --bandwidth-gb-s 8.97

import argparse
import csv
import glob
import json
import os


MODEL_LABELS = {"resnet18": "ResNet-18", "mbv2": "MobileNetV2"}
MODEL_ORDER = {"resnet18": 0, "mbv2": 1}
MODEL_ALIASES = {"mobilenetv2": "mbv2"}

SUMMARY_COLUMNS = [
    "config", "model", "pruning", "precision", "num_layers",
    "compute_bound_layers", "memory_bound_layers", "compute_bound_tau_ms",
    "memory_bound_tau_ms", "layerwise_tau_ms", "compute_bound_share_pct",
    "memory_bound_share_pct", "aggregate_tau_ms", "layerwise_to_aggregate_ratio",
    "top_layer", "top_layer_bound", "top_layer_tau_ms",
]

DETAIL_COLUMNS = [
    "config", "model", "pruning", "precision", "layer", "type", "macs",
    "mem_bytes", "arithmetic_intensity", "tau_compute_ms", "tau_memory_ms",
    "tau_layer_ms", "bound", "layerwise_share_pct",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_name(name):
    parts = name.split("_")
    try:
        index = parts.index("c10")
    except ValueError:
        raise ValueError("unexpected configuration name: %s" % name)
    model = "_".join(parts[:index])
    model = MODEL_ALIASES.get(model, model)
    pruning = parts[index + 1]
    if model not in MODEL_LABELS or not pruning.startswith("p"):
        raise ValueError("unexpected configuration name: %s" % name)
    return model, pruning


def precision_from_name(name):
    if "_fp32_" in name:
        return "FP32"
    if "_int8_" in name:
        return "INT8"
    raise ValueError("precision missing from configuration name: %s" % name)


def analyze_metric(data, path, peak_gmac_s, bandwidth_gb_s):
    name = data.get("name")
    layers = data.get("per_layer")
    if not name or not isinstance(layers, list) or not layers:
        raise ValueError("missing layer records in %s" % path)

    model_key, pruning = parse_name(name)
    precision = precision_from_name(name)
    details = []
    for layer in layers:
        macs = layer["macs"]
        mem_bytes = layer["mem_bytes"]
        if mem_bytes <= 0:
            raise ValueError("non-positive memory traffic in %s" % path)
        tau_compute = macs / (peak_gmac_s * 1e9) * 1e3
        tau_memory = mem_bytes / (bandwidth_gb_s * 1e9) * 1e3
        tau_layer = max(tau_compute, tau_memory)
        details.append({
            "config": "%s_c10_%s" % (model_key, pruning),
            "model": MODEL_LABELS[model_key],
            "_model_key": model_key,
            "pruning": pruning,
            "precision": precision,
            "layer": layer["name"],
            "type": layer["type"],
            "macs": macs,
            "mem_bytes": mem_bytes,
            "arithmetic_intensity": macs / mem_bytes,
            "tau_compute_ms": tau_compute,
            "tau_memory_ms": tau_memory,
            "tau_layer_ms": tau_layer,
            "bound": "compute" if tau_compute >= tau_memory else "memory",
        })

    total_macs = sum(row["macs"] for row in details)
    total_mem = sum(row["mem_bytes"] for row in details)
    if total_macs != data["macs"] or total_mem != data["mem_bytes"]:
        raise ValueError("per-layer sum mismatch in %s" % path)

    layerwise_tau = sum(row["tau_layer_ms"] for row in details)
    for row in details:
        row["layerwise_share_pct"] = row["tau_layer_ms"] / layerwise_tau * 100.0

    compute_rows = [row for row in details if row["bound"] == "compute"]
    memory_rows = [row for row in details if row["bound"] == "memory"]
    compute_tau = sum(row["tau_layer_ms"] for row in compute_rows)
    memory_tau = sum(row["tau_layer_ms"] for row in memory_rows)
    aggregate_tau = max(
        total_macs / (peak_gmac_s * 1e9) * 1e3,
        total_mem / (bandwidth_gb_s * 1e9) * 1e3,
    )
    top = max(details, key=lambda row: row["tau_layer_ms"])
    summary = {
        "config": details[0]["config"],
        "model": MODEL_LABELS[model_key],
        "_model_key": model_key,
        "pruning": pruning,
        "precision": precision,
        "num_layers": len(details),
        "compute_bound_layers": len(compute_rows),
        "memory_bound_layers": len(memory_rows),
        "compute_bound_tau_ms": compute_tau,
        "memory_bound_tau_ms": memory_tau,
        "layerwise_tau_ms": layerwise_tau,
        "compute_bound_share_pct": compute_tau / layerwise_tau * 100.0,
        "memory_bound_share_pct": memory_tau / layerwise_tau * 100.0,
        "aggregate_tau_ms": aggregate_tau,
        "layerwise_to_aggregate_ratio": layerwise_tau / aggregate_tau,
        "top_layer": top["layer"],
        "top_layer_bound": top["bound"],
        "top_layer_tau_ms": top["tau_layer_ms"],
    }
    return summary, details


def write_csv(rows, columns, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path, peak_gmac_s, bandwidth_gb_s):
    lines = [
        "# Layer-level roofline time-constant decomposition",
        "",
        "Calibration: peak throughput = %.2f GMAC/s; memory bandwidth = %.2f GB/s." % (
            peak_gmac_s, bandwidth_gb_s),
        "",
        "Layerwise tau is the sum over layers of max(tau_compute_layer, "
        "tau_memory_layer). It is intentionally reported separately from the "
        "aggregate roofline lower bound, because CNN layers execute serially "
        "and may have different bottlenecks.",
        "",
        "| Configuration | Precision | Compute layers | Memory layers | "
        "Compute tau share | Memory tau share | Layerwise tau (ms) | "
        "Aggregate tau (ms) | Ratio | Top layer | Top-layer tau (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {precision} | {compute_bound_layers} | "
            "{memory_bound_layers} | {compute_bound_share_pct:.1f}% | "
            "{memory_bound_share_pct:.1f}% | {layerwise_tau_ms:.3f} | "
            "{aggregate_tau_ms:.3f} | {layerwise_to_aggregate_ratio:.2f}x | "
            "{top_layer} ({top_layer_bound}) | {top_layer_tau_ms:.3f} |".format(**row))
    lines.append("")
    with open(path, "w", newline="") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--peak-gmac-s", type=float, default=24.36)
    parser.add_argument("--bandwidth-gb-s", type=float, default=8.97)
    parser.add_argument("--expected-count", type=int, default=16)
    args = parser.parse_args()

    if args.peak_gmac_s <= 0 or args.bandwidth_gb_s <= 0:
        raise SystemExit("roof parameters must be positive")

    patterns = [
        os.path.join(args.metrics_dir, "*_fp32_s42.json"),
        os.path.join(args.metrics_dir, "*_int8_s42.json"),
    ]
    paths = sorted(set(path for pattern in patterns for path in glob.glob(pattern)))
    if len(paths) != args.expected_count:
        raise SystemExit("expected %d metric files, found %d" % (
            args.expected_count, len(paths)))

    summaries = []
    details = []
    for path in paths:
        summary, per_layer = analyze_metric(
            load_json(path), path, args.peak_gmac_s, args.bandwidth_gb_s)
        summaries.append(summary)
        details.extend(per_layer)

    summaries.sort(key=lambda row: (
        MODEL_ORDER[row["_model_key"]], int(row["pruning"][1:]),
        row["precision"] != "FP32"))
    details.sort(key=lambda row: (
        MODEL_ORDER[row["_model_key"]], int(row["pruning"][1:]),
        row["precision"] != "FP32", -row["tau_layer_ms"]))

    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, "tau_layer_summary.csv")
    detail_csv = os.path.join(args.out_dir, "tau_layer_detail.csv")
    markdown = os.path.join(args.out_dir, "tau_layer_summary.md")
    write_csv(summaries, SUMMARY_COLUMNS, summary_csv)
    write_csv(details, DETAIL_COLUMNS, detail_csv)
    write_markdown(summaries, markdown, args.peak_gmac_s, args.bandwidth_gb_s)

    print("validated configurations: %d" % len(summaries))
    print("validated layer records : %d" % len(details))
    print("summary csv : %s" % summary_csv)
    print("detail csv  : %s" % detail_csv)
    print("markdown    : %s" % markdown)


if __name__ == "__main__":
    main()
