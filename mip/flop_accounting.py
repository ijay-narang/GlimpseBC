"""Job-level accounting of policy arithmetic, without replaying live models."""

from __future__ import annotations

import functools
import inspect
import json
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import loguru
import torch

from mip.flops import ExecutedFlopCounter


def _signature(value):
    if isinstance(value, torch.Tensor):
        return (
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
            value.requires_grad,
        )
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return tuple((str(k), _signature(v)) for k, v in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_signature(v) for v in value)
    return (type(value).__name__, repr(value))


def _empty_totals():
    return {
        "calls": 0,
        "examples": 0,
        "counted_flops": 0,
        "backward_flops": 0,
        "measured_calls": 0,
        "cached_calls": 0,
        "reference_calls": 0,
        "unmeasured_calls": 0,
        "failed_calls": 0,
    }


class FlopLedger:
    """Count representative calls and sum shape-matched estimates for this job.

    A multiply and an add count as two FLOPs. These are arithmetic estimates,
    not a measurement of GPU instructions, elapsed time, or simulator work.
    Unsupported operations remain visible in the accompanying JSON report.
    """

    def __init__(self, config=None, *, enabled=None):
        self.enabled = (
            os.environ.get("MIP_MEASURE_FLOPS", "1") != "0"
            if enabled is None
            else bool(enabled)
        )
        self.config = config
        self.totals = {"train": _empty_totals(), "eval": _empty_totals()}
        self.profiles = OrderedDict()
        self._cache = OrderedDict()
        self._warned = set()
        self.started_at = time.time()
        self.job_id = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
        self.profile_interval = max(
            0, int(os.environ.get("MIP_FLOP_PROFILE_INTERVAL", "1000"))
        )

    def snapshot(self):
        return {phase: dict(values) for phase, values in self.totals.items()}

    def refresh_eval(self):
        """Measure fresh representatives during each evaluation invocation."""
        for key in list(self._cache):
            if key[0] == "eval":
                del self._cache[key]

    def _warning(self, message):
        if message not in self._warned:
            self._warned.add(message)
            loguru.logger.warning(message)

    def run(
        self, phase, fn, *, inputs, batch_size, metadata=(), compiled=False, probe=None
    ):
        """Execute ``fn`` exactly once; preserve its return value and exceptions.

        For compiled calls, ``probe`` must return a counter snapshot obtained
        from an isolated reference model. The compiled call itself is untouched.
        """
        if not self.enabled:
            return fn()
        key = (phase, _signature(inputs), repr(metadata), bool(compiled))
        totals = self.totals[phase]
        entry = self._cache.get(key)
        refresh = entry is None or (
            self.profile_interval > 0
            and totals["calls"] - entry["at_call"] >= self.profile_interval
        )
        report = None if refresh else entry["report"]
        source = "cached"
        counter = None
        if refresh:
            source = "reference" if compiled else "measured"
            if compiled:
                try:
                    if probe is None:
                        raise RuntimeError("no isolated reference profiler supplied")
                    report = probe()
                except Exception as exc:
                    report = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                try:
                    counter = ExecutedFlopCounter()
                    counter.__enter__()
                except Exception as exc:
                    counter = None
                    report = {"error": f"{type(exc).__name__}: {exc}"}
        work_error = None
        try:
            result = fn()
        except BaseException:
            work_error = sys.exc_info()
            totals["failed_calls"] += 1
            raise
        finally:
            if counter is not None:
                try:
                    counter.__exit__(*(work_error or (None, None, None)))
                    if work_error is None:
                        report = counter.snapshot()
                except Exception as exc:
                    report = {"error": f"{type(exc).__name__}: {exc}"}
                    self._warning(f"FLOP profiler cleanup failed: {exc}")
            if work_error is not None:
                self.write_report()
        if report and "counted_flops" in report:
            try:
                report["counted_flops"] = int(report["counted_flops"])
                report["backward_flops"] = int(report.get("backward_flops", 0))
                if min(report["counted_flops"], report["backward_flops"]) < 0:
                    raise ValueError("negative FLOP count")
            except (ValueError, OverflowError, TypeError) as exc:
                report = {"error": f"invalid FLOP profile: {exc}"}
        if refresh:
            entry = {"report": report, "at_call": totals["calls"]}
            self._cache[key] = entry
            self._cache.move_to_end(key)
            if len(self._cache) > 128:
                self._cache.popitem(last=False)
            profile_key = repr(key)
            previous = self.profiles.get(profile_key, {})
            self.profiles[profile_key] = {
                "phase": phase,
                "batch_size": int(batch_size),
                "signature": repr(key[1]),
                "metadata": repr(metadata),
                "method": "isolated_cpu_reference"
                if compiled
                else "observed_eager_call",
                "measurements": previous.get("measurements", 0) + 1,
                "report": report,
            }
            if report and "counted_flops" in report:
                loguru.logger.info(
                    f"FLOPs {phase}: {report['counted_flops'] / 1e9:.4f} GFLOPs/batch "
                    f"({int(batch_size)} examples, {source}; arithmetic estimate)"
                )
        totals["calls"] += 1
        totals["examples"] += int(batch_size)
        if not report or "counted_flops" not in report:
            totals["unmeasured_calls"] += 1
            self._warning(f"FLOP accounting incomplete for {phase}: {report}")
        else:
            totals["counted_flops"] += int(report["counted_flops"])
            totals["backward_flops"] += int(report.get("backward_flops", 0))
            totals[f"{source}_calls"] += 1
        return result

    def metrics(self, *, before=None, prefix="flops/job"):
        if not self.enabled:
            return {}
        metrics = {}
        grand_total = 0
        for phase, current in self.totals.items():
            old = _empty_totals() if before is None else before[phase]
            delta = {key: value - old[key] for key, value in current.items()}
            base = f"{prefix}/{phase}"
            calls, examples, count = (
                delta["calls"],
                delta["examples"],
                delta["counted_flops"],
            )
            for key in (
                "calls",
                "examples",
                "measured_calls",
                "cached_calls",
                "reference_calls",
                "unmeasured_calls",
                "failed_calls",
            ):
                metrics[f"{base}/{key}"] = delta[key]
            metrics[f"{base}/gflops"] = count / 1e9
            metrics[f"{base}/backward_gflops"] = delta["backward_flops"] / 1e9
            metrics[f"{base}/gflops_per_call"] = count / max(calls, 1) / 1e9
            metrics[f"{base}/gflops_per_example"] = count / max(examples, 1) / 1e9
            grand_total += count
        metrics[f"{prefix}/gflops"] = grand_total / 1e9
        return metrics

    def write_report(self):
        if not self.enabled or self.config is None:
            return
        log_config = getattr(self.config, "log", None)
        directory = getattr(log_config, "log_dir", None)
        if not directory:
            return
        report = {
            "version": 1,
            "job_id": self.job_id,
            "started_at": self.started_at,
            "scope": "current process/job only; prior checkpoint work is excluded",
            "failed_call_scope": "Totals include completed calls only; arithmetic before an interrupted or failed call is excluded",
            "convention": "2 FLOPs per multiply-add; counted arithmetic estimate",
            "coverage": "Partial: consult each profile's unsupported operations.",
            "training_scope": "Policy forward, loss, backward, optimizer and EMA operations supported by the counter",
            "evaluation_scope": "Actual policy sample calls, including all sampling iterations; excludes environment/data/video work",
            "totals_method": "Representative operation counts reused for matching shapes/settings; fresh representatives each evaluation and periodically during training",
            "totals": self.snapshot(),
            "profiles": list(self.profiles.values()),
        }
        try:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(report, indent=2, allow_nan=False)
            # Keep per-job reports when a checkpoint is resumed in the same directory.
            for name in ("flops.json", f"flops_{self.job_id}.json"):
                target = root / name
                temporary = target.with_suffix(f".tmp.{os.getpid()}")
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, target)
        except (OSError, TypeError, ValueError) as exc:
            self._warning(f"Could not write FLOP report: {exc}")


def get_job_flop_metrics(agent):
    ledger = getattr(agent, "flops", None)
    if ledger is None:
        return {}
    ledger.write_report()
    return ledger.metrics()


def report_job_flops(agent):
    metrics = get_job_flop_metrics(agent)
    if metrics:
        loguru.logger.info(
            f"Job counted arithmetic: {metrics['flops/job/gflops']:.4f} GFLOPs; details in flops.json"
        )
    return metrics


def track_evaluation_flops(scope):
    """Attach fresh per-evaluation statistics without changing evaluation work."""

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            agent = arguments.arguments["agent"]
            ledger = getattr(agent, "flops", None)
            if ledger is None or not ledger.enabled:
                return fn(*args, **kwargs)
            before = ledger.snapshot()
            ledger.refresh_eval()
            try:
                result = fn(*args, **kwargs)
            finally:
                ledger.write_report()
            num_steps = arguments.arguments.get("num_steps", 1)
            interval = ledger.metrics(before=before, prefix=f"flops/eval/{scope}")
            result.update(
                {f"{key}_{num_steps}": value for key, value in interval.items()}
            )
            return result

        return wrapped

    return decorate
