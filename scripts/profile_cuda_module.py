#!/usr/bin/env python3
"""Run a Python module while recording wall time and peak CUDA memory."""

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-images", type=int, required=True)
    args, module_args = parser.parse_known_args()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    sys.argv = [args.module, *module_args]
    runpy.run_module(args.module, run_name="__main__")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    result = {
        "module": args.module,
        "num_images": args.num_images,
        "elapsed_seconds": elapsed,
        "images_per_second": args.num_images / elapsed,
        "seconds_per_image": elapsed / args.num_images,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device": torch.cuda.get_device_name(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
