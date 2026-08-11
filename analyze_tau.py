#!/usr/bin/env python3
# Compute roofline time constants from the validated experiment summary.
#
# Usage:
#   python analyze_tau.py --input results/fp32_int8_summary.csv \
#       --out-dir results --peak-gmac-s 24.36 --bandwidth-gb-s 8.97

import argparse
import csv
import os


MODEL_ORDER = {"ResNet-18": 0, "MobileNetV2": 1}

CSV_COLUMNS = [
    "config", "model", "pruning", "precision", "accuracy_pct", "params",
    "macs", "memory_bytes", "arithmetic_intensity", "tau_compute_ms",
    "tau_memory_ms", "tau_theory_ms", "bound", "measured_latency_ms",
    "gap_ratio", "achieved_gmac_s", "roof_utilization_pct",
]


def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid %s in %s" % (key, row.get("config", "row"))) from exc


def make_row(source, precision, peak_gmac_s, bandwidth_gb_s):
    prefix = precision + "_"
    macs = as_float(source, "macs")
    memory_bytes = as_float(source, prefix + "mem_bytes")
    ai = as_float(source, prefix + "arithmetic_intensity")
    latency = as_float(source, prefix + "latency_ms")
    accuracy = as_float(source, prefix + "accuracy_pct")

    tau_compute_ms = macs / (peak_gmac_s * 1e9) * 1e3
    tau_memory_ms = memory_bytes / (bandwidth_gb_s * 1e9) * 1e3
    tau_theory_ms = max(tau_compute_ms, tau_memory_ms)
    bound = "compute" if tau_compute_ms >= tau_memory_ms else "memory"
    achieved_gmac_s = macs / latency / 1e6

    return {
        "config": source["config"],
        "model": source["model"],
        "pruning": source["pruning"],
        "precision": precision.upper(),
        "accuracy_pct": accuracy,
        "params": int(as_float(source, "params")),
        "macs": int(macs),
        "memory_bytes": int(memory_bytes),
        "arithmetic_intensity": ai,
        "tau_compute_ms": tau_compute_ms,
        "tau_memory_ms": tau_memory_ms,
        "tau_theory_ms": tau_theory_ms,
        "bound": bound,
        "measured_latency_ms": latency,
        "gap_ratio": latency / tau_theory_ms,
        "achieved_gmac_s": achieved_gmac_s,
        "roof_utilization_pct": achieved_gmac_s / peak_gmac_s * 100.0,
    }


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path, peak_gmac_s, bandwidth_gb_s):
    ridge = peak_gmac_s / bandwidth_gb_s
    lines = [
        "# Roofline time-constant analysis",
        "",
        "Calibration: single-thread peak throughput = %.2f GMAC/s; memory "
        "bandwidth = %.2f GB/s; ridge point = %.3f MACs/Byte." % (
            peak_gmac_s, bandwidth_gb_s, ridge),
        "",
        "For each row, tau_compute = MACs / peak throughput, tau_memory = "
        "memory bytes / bandwidth, and tau_theory = max(tau_compute, "
        "tau_memory). The gap is measured latency / tau_theory.",
        "",
        "INT8 rows use the same measured roof as a normalized reference for "
        "comparing data movement. They are not an independent measurement of "
        "the processor's INT8 peak throughput; their observed latency must "
        "therefore be interpreted together with the fbgemm proxy measurement.",
        "",
        "| Configuration | Precision | AI | Bound | tau_compute (ms) | "
        "tau_memory (ms) | tau_theory (ms) | Measured (ms) | Gap | "
        "Achieved (GMAC/s) | Roof util. |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {precision} | {arithmetic_intensity:.3f} | {bound} | "
            "{tau_compute_ms:.3f} | {tau_memory_ms:.3f} | "
            "{tau_theory_ms:.3f} | {measured_latency_ms:.3f} | "
            "{gap_ratio:.3f}x | {achieved_gmac_s:.3f} | "
            "{roof_utilization_pct:.1f}% |".format(**row))
    lines.append("")
    with open(path, "w", newline="") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/fp32_int8_summary.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--peak-gmac-s", type=float, default=24.36)
    parser.add_argument("--bandwidth-gb-s", type=float, default=8.97)
    args = parser.parse_args()

    if args.peak_gmac_s <= 0 or args.bandwidth_gb_s <= 0:
        raise SystemExit("roof parameters must be positive")

    with open(args.input, newline="") as f:
        source_rows = list(csv.DictReader(f))
    if len(source_rows) != 8:
        raise SystemExit("expected 8 configuration pairs, found %d" % len(source_rows))

    rows = []
    for source in source_rows:
        rows.append(make_row(source, "fp32", args.peak_gmac_s, args.bandwidth_gb_s))
        rows.append(make_row(source, "int8", args.peak_gmac_s, args.bandwidth_gb_s))

    rows.sort(key=lambda r: (
        MODEL_ORDER[r["model"]], int(r["pruning"][1:]), r["precision"] != "FP32"))
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "tau_analysis.csv")
    markdown_path = os.path.join(args.out_dir, "tau_analysis.md")
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path, args.peak_gmac_s, args.bandwidth_gb_s)

    print("validated rows: %d" % len(rows))
    print("csv     : %s" % csv_path)
    print("markdown: %s" % markdown_path)


if __name__ == "__main__":
    main()
