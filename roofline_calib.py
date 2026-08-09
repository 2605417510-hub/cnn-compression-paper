#!/usr/bin/env python3
# Roofline roof calibration.
#
# The roofline needs two numbers:
#   peak_throughput  -> the height of the horizontal roof   [MAC/s]
#   memory_bandwidth -> the slope of the diagonal roof      [byte/s]
#
# WHY NOT JUST READ THE SPEC SHEET
#   Spec figures describe all-core peak compute and all-channel peak
#   bandwidth. This study measures latency single-threaded at batch size 1.
#   One thread occupies one core and cannot sustain enough outstanding
#   misses to approach multi-channel bandwidth. Using spec numbers as the
#   roof would put every configuration far below it and the plot would
#   carry no information. So both numbers are measured under exactly the
#   conditions the latency protocol uses: one thread.
#
# HOW EACH IS MEASURED
#   peak throughput  large square matmul. GEMM is the densest MAC workload
#                    available and reaches close to the practical ceiling.
#                    An N x N by N x N matmul performs N^3 MACs.
#   bandwidth        copy of an array far larger than last-level cache, so
#                    every element really comes from DRAM. Counts 2 bytes of
#                    traffic per element moved: one read plus one write.
#
# Usage:
#   python roofline_calib.py
#   python roofline_calib.py --out results/roofline_roof.json
#   python roofline_calib.py --matmul-n 3072 --array-mb 512 --repeats 7

import argparse
import json
import os
import platform
import statistics
import time

import torch


def measure_peak_throughput(n, repeats):
    """Single-thread dense matmul. Returns MAC/s."""
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    macs = float(n) ** 3          # n^3 multiply-accumulates

    for _ in range(2):            # warmup
        torch.matmul(a, b)

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        torch.matmul(a, b)
        samples.append(time.perf_counter() - t0)

    best = min(samples)           # best case = closest to the true ceiling
    median = statistics.median(samples)
    return {
        "matmul_n": n,
        "macs_per_call": macs,
        "seconds_best": best,
        "seconds_median": median,
        "peak_mac_per_s": macs / best,
        "peak_gmac_per_s": macs / best / 1e9,
        "peak_gflop_per_s": 2.0 * macs / best / 1e9,
    }


def measure_bandwidth(array_mb, repeats):
    """Single-thread DRAM copy. Returns byte/s counting read + write."""
    n_elems = array_mb * 1024 * 1024 // 4        # float32
    src = torch.randn(n_elems)
    dst = torch.empty_like(src)
    bytes_moved = 2.0 * n_elems * 4              # read once, write once

    for _ in range(2):
        dst.copy_(src)

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        dst.copy_(src)
        samples.append(time.perf_counter() - t0)

    best = min(samples)
    median = statistics.median(samples)
    return {
        "array_mb": array_mb,
        "bytes_per_call": bytes_moved,
        "seconds_best": best,
        "seconds_median": median,
        "bandwidth_byte_per_s": bytes_moved / best,
        "bandwidth_gb_per_s": bytes_moved / best / 1e9,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matmul-n", type=int, default=2048)
    p.add_argument("--array-mb", type=int, default=256,
                   help="must be far larger than last-level cache")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.set_num_threads(1)      # same condition as the latency protocol

    print("=" * 62)
    print("roofline roof calibration   (single thread)")
    print("=" * 62)

    comp = measure_peak_throughput(args.matmul_n, args.repeats)
    print("peak throughput")
    print("  matmul        : %d x %d" % (args.matmul_n, args.matmul_n))
    print("  MACs per call : %.3e" % comp["macs_per_call"])
    print("  best time     : %.4f s" % comp["seconds_best"])
    print("  -> peak       : %8.2f GMAC/s   (%.2f GFLOP/s)"
          % (comp["peak_gmac_per_s"], comp["peak_gflop_per_s"]))
    print("-" * 62)

    mem = measure_bandwidth(args.array_mb, args.repeats)
    print("memory bandwidth")
    print("  array         : %d MB (float32)" % args.array_mb)
    print("  bytes moved   : %.3e  (read + write)" % mem["bytes_per_call"])
    print("  best time     : %.4f s" % mem["seconds_best"])
    print("  -> bandwidth  : %8.2f GB/s" % mem["bandwidth_gb_per_s"])
    print("=" * 62)

    peak = comp["peak_mac_per_s"]
    bw = mem["bandwidth_byte_per_s"]
    ridge = peak / bw
    print("ridge point   : %.3f MACs/Byte" % ridge)
    print("  models with arithmetic intensity above this are compute bound,")
    print("  below it, memory bound.")
    print("=" * 62)

    # sanity: the roof must sit above anything already measured
    print("sanity check")
    print("  ResNet-18 measured 555422720 MACs in 33.372 ms -> 16.64 GMAC/s")
    if comp["peak_gmac_per_s"] > 16.64:
        print("  peak (%.2f) exceeds it   OK" % comp["peak_gmac_per_s"])
    else:
        print("  peak (%.2f) is BELOW it   PROBLEM" % comp["peak_gmac_per_s"])
        print("  a roof beneath a measured point means the calibration is")
        print("  wrong, or something else was competing for the CPU.")

    result = {
        "compute": comp,
        "memory": mem,
        "ridge_point_mac_per_byte": ridge,
        "threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("written : %s" % args.out)


if __name__ == "__main__":
    main()
