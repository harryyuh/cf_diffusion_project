"""Opt-in optimizer-step timer for isolated training benchmark processes."""

import json
import os
import time


if os.environ.get("CF_STEP_PROFILE_OUT"):
    import torch

    _original_step = torch.optim.AdamW.step
    _count = 0
    _start = None
    _warmup = int(os.environ.get("CF_STEP_PROFILE_WARMUP", "30"))
    _measure = int(os.environ.get("CF_STEP_PROFILE_MEASURE", "200"))

    def _timed_step(self, *args, **kwargs):
        global _count, _start
        result = _original_step(self, *args, **kwargs)
        _count += 1
        if _count == _warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            _start = time.perf_counter()
        elif _count == _warmup + _measure:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - _start
            payload = {
                "method": os.environ.get("CF_STEP_PROFILE_METHOD"),
                "repeat": int(os.environ.get("CF_STEP_PROFILE_REPEAT", "0")),
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")),
                "warmup_steps": _warmup,
                "measured_steps": _measure,
                "batch_size": 40,
                "precision": "fp32",
                "elapsed_seconds": elapsed,
                "seconds_per_step": elapsed / _measure,
                "images_per_second": 40 * _measure / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "torch_version": torch.__version__,
            }
            with open(os.environ["CF_STEP_PROFILE_OUT"], "w") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            raise SystemExit(0)
        return result

    torch.optim.AdamW.step = _timed_step
