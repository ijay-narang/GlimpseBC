#!/usr/bin/env python3
"""Plan and submit immutable offline-validation annealing sweeps.

This module intentionally uses only the Python standard library so planning and
submission do not import Torch, simulators, or benchmark datasets.  The Slurm
workers call ``val_anneal/train_offline.py``; no rollout entrypoint is used.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCHMARKS = {"pusht", "libero", "robomimic"}
VARIANTS = {"vanilla", "foveated"}
VAL_FRACTION = 0.20
SPLIT_SEED = 10_000
TRAIN_SEED = 0
GRADIENT_STEPS = 200_000
ANNEAL_STEPS = 100_000
BATCH_SIZE = 32
SAVE_FREQ = 20_000
LOG_FREQ = 1_000
LIBERO_TASK_IDS = (1, 2, 3)

PLAN_COLUMNS = (
    "array_index",
    "benchmark",
    "variant",
    "task",
    "task_id",
    "resolution",
    "source_px",
    "start_px",
    "end_px",
    "patch_size",
    "start_scale",
    "radius",
    "start_radius",
    "anneal_steps",
)

WORKER = r"""#!/usr/bin/env bash
#SBATCH --job-name=val-anneal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=run_logs/slurm/slurm_%x_%A_%a.out
#SBATCH --error=run_logs/slurm/slurm_%x_%A_%a.err
set -euo pipefail
: "${SWEEP_ROOT:?SWEEP_ROOT must be exported by the submitter}"
: "${REPO_ROOT:?REPO_ROOT must be exported by the submitter}"
: "${SLURM_ARRAY_TASK_ID:?This worker must run as a Slurm array task}"
[[ $# -le 1 ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }
case "${1:-}" in
  --dry-run) action=--print-row ;;
  '') action=--run-row ;;
  *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
cd "${REPO_ROOT}"
source "${SWEEP_ROOT}/settings.sh"
exec "${PLANNER_PYTHON}" "${SWEEP_ROOT}/launchers/val_anneal/submit_grid.py" \
  "${action}" "${REPO_ROOT}" "${SWEEP_ROOT}" "${SLURM_ARRAY_TASK_ID}"
"""


def fail(message: str) -> None:
    raise SystemExit(message)


def _as_posix(path: Path | str) -> str:
    return Path(path).as_posix()


def _format_number(value: float | int) -> str:
    return format(value, ".12g")


def _write_tsv(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=columns, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def keep_exact(path: Path, content: bytes | str) -> None:
    """Create one immutable artifact, allowing byte-identical repeats."""
    payload = content.encode("utf-8") if isinstance(content, str) else content
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            fail(f"Existing artifact differs; choose a new SWEEP_ID: {path}")


def grid(benchmark: str, variant: str) -> list[dict[str, Any]]:
    """Return the frozen science grid for one benchmark/encoder pair."""
    if benchmark not in BENCHMARKS:
        fail(f"Unknown benchmark {benchmark!r}")
    if variant not in VARIANTS:
        fail(f"Unknown variant {variant!r}")

    rows: list[dict[str, Any]] = []
    if benchmark == "pusht":
        tasks = [("pusht_image", "-", "96x96", 96)]
        pairs = [
            (start, end)
            for start in (48, 72, 96)
            for end in (24, 32, 48, 64)
            if start > end
        ]
    elif benchmark == "libero":
        tasks = [
            ("libero_spatial_0_image", str(task_id), "128x128", 128)
            for task_id in LIBERO_TASK_IDS
        ]
        pairs = [
            (start, end)
            for start in (64, 96, 128)
            for end in (32, 48, 64, 96)
            if start > end
        ]
    else:
        tasks = [
            (task, "-", "84x84", 84)
            for task in (
                "can_ph_image",
                "lift_ph_image",
                "square_ph_image",
                "tool_hang_ph_image",
            )
        ]
        # Use only the native 84x84 streams declared by the main task configs.
        pairs = None

    if benchmark == "libero":
        # Keep one contiguous task-ID block per geometry.  This validation-only
        # sweep intentionally covers LIBERO Spatial task IDs 1, 2, and 3.
        assert pairs is not None
        combinations = [
            (task, task_id, resolution, source_px, start_px, end_px)
            for start_px, end_px in pairs
            for task, task_id, resolution, source_px in tasks
        ]
    elif benchmark == "robomimic":
        combinations = [
            (task, task_id, resolution, source_px, source_px, end_px)
            for task, task_id, resolution, source_px in tasks
            for end_px in (32, 48, 64)
        ]
    else:
        assert pairs is not None
        combinations = [
            (task, task_id, resolution, source_px, start_px, end_px)
            for start_px, end_px in pairs
            for task, task_id, resolution, source_px in tasks
        ]

    for task, task_id, resolution, source_px, start_px, end_px in combinations:
        if not 0 < end_px < start_px <= source_px:
            fail(
                f"Invalid annealing geometry for {benchmark}: "
                f"source={source_px}, start={start_px}, end={end_px}"
            )
        rows.append(
            {
                "array_index": len(rows),
                "benchmark": benchmark,
                "variant": variant,
                "task": task,
                "task_id": task_id,
                "resolution": resolution,
                "source_px": source_px,
                "start_px": start_px,
                "end_px": end_px,
                "patch_size": end_px,
                "start_scale": _format_number(start_px / source_px),
                "radius": _format_number(end_px / 2),
                "start_radius": _format_number(start_px / 2),
                "anneal_steps": ANNEAL_STEPS,
            }
        )

    expected = {"pusht": 10, "libero": 27, "robomimic": 12}[benchmark]
    if len(rows) != expected:
        fail(f"Internal grid error: expected {expected} rows, produced {len(rows)}")
    return rows


def settings(
    root: Path, benchmark: str, variant: str, max_concurrent: int
) -> dict[str, Any]:
    """Freeze science settings while accepting deployment-path overrides."""
    defaults: dict[str, Path] = {
        "PLANNER_PYTHON": root / ".venv/bin/python",
        "PUSHT_PYTHON_BIN": root / ".venv/bin/python",
        "LIBERO_PYTHON_BIN": root / ".venv-libero/bin/python",
        "ROBOMIMIC_PYTHON_BIN": root / ".venv/bin/python",
        "LIBERO_DATA_ROOT": root / "libero_datasets",
        "HF_HOME": root / ".cache/hf_cache",
    }
    values: dict[str, Any] = {
        "BENCHMARK": benchmark,
        "VARIANT": variant,
        "MAX_CONCURRENT": max_concurrent,
        "VAL_FRACTION": _format_number(VAL_FRACTION),
        "SPLIT_SEED": SPLIT_SEED,
        "TRAIN_SEED": TRAIN_SEED,
        "GRADIENT_STEPS": GRADIENT_STEPS,
        "ANNEAL_STEPS": ANNEAL_STEPS,
        "BATCH_SIZE": BATCH_SIZE,
        "SAVE_FREQ": SAVE_FREQ,
        "LOG_FREQ": LOG_FREQ,
        "NETWORK": "rnn",
        "LOSS_TYPE": "regression",
        "FOVEA_RATIO": _format_number(0.4),
        "BLUR_SIGMA": _format_number(2.0),
        "MIP_MEASURE_FLOPS": 1,
        "PYTHONHASHSEED": TRAIN_SEED,
        "PYTHONPATH": _as_posix(root),
    }
    for key, default in defaults.items():
        supplied = os.environ.get(key)
        path = Path(supplied).expanduser() if supplied else default
        # Keep venv paths intact; resolving a symlink can escape its pyvenv.cfg.
        values[key] = _as_posix(Path(os.path.abspath(path)))
    pusht_data_path = os.environ.get("PUSHT_DATA_PATH", "").strip()
    values["PUSHT_DATA_PATH"] = (
        _as_posix(Path(os.path.abspath(Path(pusht_data_path).expanduser())))
        if pusht_data_path
        else ""
    )
    hf_home = Path(str(values["HF_HOME"]))
    values.update(
        {
            "HUGGINGFACE_HUB_CACHE": _as_posix(hf_home / "hub"),
            "HF_XET_CACHE": _as_posix(hf_home / "xet"),
            "UV_CACHE_DIR": _as_posix(root / ".cache/uv_cache"),
            "XDG_CACHE_HOME": _as_posix(root / ".cache"),
            "TMPDIR": _as_posix(root / ".cache/tmp"),
            "LIBERO_CONFIG_PATH": _as_posix(root / ".cache/libero_config"),
        }
    )
    return values


def _run_id(row: dict[str, Any]) -> str:
    benchmark = str(row["benchmark"])
    variant = str(row["variant"])
    geometry = f"s{row['start_px']}_e{row['end_px']}"
    if benchmark == "pusht":
        subject = "pusht"
    elif benchmark == "libero":
        subject = f"libero_spatial_task{row['task_id']}"
    else:
        subject = f"{row['task']}_{row['resolution']}"
    return (
        f"{subject}_annealed_{variant}_{geometry}_rnn_regression_"
        f"seed{TRAIN_SEED}_split{SPLIT_SEED}_val20"
    )


def command_for(
    root: Path, sweep_root: Path, row: dict[str, Any], values: dict[str, Any]
) -> list[str]:
    """Compose one exact offline-training argv without rollout arguments."""
    benchmark = str(row["benchmark"])
    variant = str(row["variant"])
    python_key = {
        "pusht": "PUSHT_PYTHON_BIN",
        "libero": "LIBERO_PYTHON_BIN",
        "robomimic": "ROBOMIMIC_PYTHON_BIN",
    }[benchmark]
    run_id = _run_id(row)
    run_log_dir = sweep_root / benchmark / "logs" / run_id
    argv = [
        str(values[python_key]),
        _as_posix(root / "val_anneal/train_offline.py"),
        f"task={row['task']}",
        "network=rnn",
        f"task.crop_shape=[{row['patch_size']},{row['patch_size']}]",
        "optimization.loss_type=regression",
        f"optimization.batch_size={BATCH_SIZE}",
        f"optimization.gradient_steps={GRADIENT_STEPS}",
        f"optimization.seed={TRAIN_SEED}",
        "optimization.use_compile=false",
        "optimization.use_cudagraphs=false",
        "optimization.auto_resume=false",
        f"log.save_freq={SAVE_FREQ}",
        f"log.log_freq={LOG_FREQ}",
        f"log.log_dir={_as_posix(run_log_dir)}",
        f"log.exp_name={run_id}",
        f"log.group=val_anneal_{benchmark}_{variant}",
        f"+validation.benchmark={benchmark}",
        f"+validation.fraction={_format_number(VAL_FRACTION)}",
        f"+validation.split_seed={SPLIT_SEED}",
    ]

    if variant == "vanilla":
        argv.extend(
            [
                "network.encoder_type=annealed_vanilla_saccade_image",
                f"network.annealed_vanilla_patch_size={row['patch_size']}",
                f"network.annealed_vanilla_start_scale={row['start_scale']}",
                f"network.annealed_vanilla_anneal_steps={ANNEAL_STEPS}",
            ]
        )
    else:
        argv.extend(
            [
                "network.encoder_type=annealed_foveated_saccade_image",
                f"network.foveated_patch_size={row['patch_size']}",
                f"network.foveated_radius={row['radius']}",
                f"network.foveated_start_radius={row['start_radius']}",
                "network.foveated_fovea_ratio=0.4",
                "network.foveated_blur_sigma=2",
                f"network.foveated_anneal_steps={ANNEAL_STEPS}",
            ]
        )

    if benchmark == "libero":
        argv.extend(
            [
                "task.libero_suite=libero_spatial",
                "task.env_type=libero_spatial",
                f"task.libero_task_id={row['task_id']}",
                f"task.dataset_root={values['LIBERO_DATA_ROOT']}",
            ]
        )
    elif benchmark == "pusht" and values.get("PUSHT_DATA_PATH"):
        argv.extend(
            [
                "task.dataset_repo=null",
                "task.dataset_filename=null",
                f"+task.dataset_path={values['PUSHT_DATA_PATH']}",
            ]
        )
    return argv


def _relevant_source(path: Path) -> bool:
    parts = path.parts
    if not parts:
        return False
    if path.as_posix() in {"pyproject.toml", "uv.lock"}:
        return True
    if parts[0] == "mip":
        return path.suffix == ".py"
    if parts[0] == "examples":
        return path.suffix in {".py", ".yaml", ".yml"}
    if parts[0] == "val_anneal":
        return path.suffix in {".py", ".sh", ".yaml", ".yml", ".md"}
    return path.as_posix() in {
        "scripts/libero/requirements.txt",
        "scripts/libero/utils/setup_libero_env.sh",
    }


def _source_names(root: Path) -> list[Path]:
    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        names = [Path(os.fsdecode(item)) for item in raw.split(b"\0") if item]
    except (OSError, subprocess.CalledProcessError):
        names = [path.relative_to(root) for path in root.rglob("*") if path.is_file()]
    return sorted({path for path in names if _relevant_source(path)})


def capture_source(root: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    records: list[dict[str, Any]] = []
    contents: dict[str, bytes] = {}
    for relative in _source_names(root):
        source = root / relative
        if not source.is_file() or source.is_symlink():
            continue
        content = source.read_bytes()
        name = relative.as_posix()
        contents[name] = content
        records.append(
            {
                "path": name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return records, contents


def _source_manifest(records: list[dict[str, Any]]) -> bytes:
    return (json.dumps(records, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _source_archive(contents: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(contents):
            info = tarfile.TarInfo(name)
            info.size = len(contents[name])
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(contents[name]))
    return output.getvalue()


def verify_source(root: Path, sweep_root: Path) -> None:
    expected_path = sweep_root / "source_manifest.json"
    if not expected_path.is_file():
        fail(f"Missing source manifest: {expected_path}")
    current, _ = capture_source(root)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if current != expected:
        expected_map = {item["path"]: item for item in expected}
        current_map = {item["path"]: item for item in current}
        changed = sorted(
            name
            for name in expected_map.keys() | current_map.keys()
            if expected_map.get(name) != current_map.get(name)
        )
        detail = ", ".join(changed[:8])
        if len(changed) > 8:
            detail += f", ... ({len(changed)} files)"
        fail(
            "Relevant source changed after this sweep was planned; choose a new "
            f"SWEEP_ID. Changed: {detail or 'unknown'}"
        )


def _load_settings(sweep_root: Path) -> dict[str, Any]:
    return json.loads((sweep_root / "settings.json").read_text(encoding="utf-8"))


def _load_row(sweep_root: Path, index: int) -> dict[str, str]:
    values = _load_settings(sweep_root)
    benchmark = str(values["BENCHMARK"])
    rows = _read_tsv(sweep_root / "plans" / f"{benchmark}.tsv")
    if not 0 <= index < len(rows):
        fail(f"Array index {index} is outside 0..{len(rows) - 1}")
    row = rows[index]
    if int(row["array_index"]) != index:
        fail(f"Plan row/index mismatch at {index}")
    return row


def print_or_run_row(
    root: Path, sweep_root: Path, index: int, *, execute: bool
) -> None:
    values = _load_settings(sweep_root)
    row = _load_row(sweep_root, index)
    argv = command_for(root, sweep_root, row, values)
    print("Command: " + shlex.join(argv), flush=True)
    if not execute:
        return
    verify_source(root, sweep_root)
    python_path = Path(argv[0])
    if not python_path.is_file():
        fail(f"Training interpreter does not exist: {python_path}")
    run_id = _run_id(row)
    run_log_dir = sweep_root / row["benchmark"] / "logs" / run_id
    if run_log_dir.exists():
        fail(f"Run output already exists: {run_log_dir}")
    lock = sweep_root / ".run_locks" / run_id
    try:
        lock.mkdir(parents=True)
    except FileExistsError:
        fail(f"Run is already claimed or needs manual inspection: {lock}")
    (lock / "command.txt").write_text(shlex.join(argv) + "\n", encoding="utf-8")
    os.execve(argv[0], argv, dict(os.environ))


def _settings_shell(values: dict[str, Any]) -> str:
    header = "# Immutable offline-validation annealing settings.\n"
    return header + "".join(
        f"export {key}={shlex.quote(str(value))}\n" for key, value in values.items()
    )


def _wrapper_path(root: Path, benchmark: str, variant: str) -> Path:
    return root / f"val_anneal/submit_{benchmark}_annealed_{variant}.sh"


def _help(benchmark: str, variant: str, wrapper: Path) -> str:
    count = len(grid(benchmark, variant))
    return f"""Usage: bash {wrapper.as_posix()} [--dry-run|--submit|--help]

Default: --dry-run. Plans {count} {benchmark} annealed-{variant} runs as one
Slurm array. Every run uses an 80/20 demonstration/episode split, split seed
10000, train seed 0, RNN/regression, batch 32, 200000 optimizer updates, and
100000 annealing forwards. Validation is offline only; no simulator rollouts or
evaluation episodes are requested.

Deployment-only overrides:
  SWEEP_ID=<unique-name>  MAX_CONCURRENT=0  (0 means no array throttle)
  PLANNER_PYTHON=<path>  PUSHT_PYTHON_BIN=<path>
  LIBERO_PYTHON_BIN=<path>
  ROBOMIMIC_PYTHON_BIN=<path>  LIBERO_DATA_ROOT=<path>
  PUSHT_DATA_PATH=<path>  HF_HOME=<path>

Planning records exact commands, settings, plans, the worker, and a deterministic
source snapshot below run_logs/val_anneal/<SWEEP_ID>. Reusing --submit reports
the existing Slurm job. A changed plan or source requires a new SWEEP_ID.
"""


def _parse_nonnegative_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    if not re.fullmatch(r"0|[1-9][0-9]*", raw):
        fail(f"{name} must be a nonnegative integer")
    return int(raw)


def _slurm_array_spec(task_count: int, max_concurrent: int) -> str:
    """Build an array range; zero concurrency means no Slurm throttle."""
    if task_count < 1:
        fail("A Slurm array must contain at least one task")
    base = f"0-{task_count - 1}"
    return base if max_concurrent == 0 else f"{base}%{max_concurrent}"


def _acquire_lock(run_root: Path) -> Path:
    lock = run_root / ".planning.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        fail(f"Another plan/submission is active, or left a lock: {lock}")
    return lock


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable\n"


def plan(
    root: Path,
    benchmark: str,
    variant: str,
    mode: str,
    wrapper: Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    max_concurrent = _parse_nonnegative_int("MAX_CONCURRENT", "0")
    sweep_id = os.environ.get(
        "SWEEP_ID",
        f"{benchmark}_annealed_{variant}_val20_split10000_seed0",
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sweep_id):
        fail("SWEEP_ID may contain only letters, digits, underscores, and hyphens")
    run_root = root / "run_logs" / "val_anneal" / sweep_id
    run_root.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(run_root)
    try:
        rows = grid(benchmark, variant)
        values = settings(root, benchmark, variant, max_concurrent)
        records, contents = capture_source(root)
        manifest = _source_manifest(records)
        fingerprint = hashlib.sha256(manifest).hexdigest() + "\n"
        keep_exact(run_root / "settings.json", json.dumps(values, indent=2) + "\n")
        keep_exact(run_root / "settings.sh", _settings_shell(values))
        keep_exact(
            run_root / "plans" / f"{benchmark}.tsv",
            _write_tsv(PLAN_COLUMNS, rows),
        )
        keep_exact(run_root / "source_manifest.json", manifest)
        keep_exact(run_root / "source_fingerprint.sha256", fingerprint)
        keep_exact(run_root / "source_snapshot.tar", _source_archive(contents))
        keep_exact(run_root / "git_commit.txt", _git_commit(root))
        keep_exact(run_root / "submission_script.sh", wrapper.read_bytes())
        keep_exact(run_root / "launchers" / "worker.sh", WORKER)
        keep_exact(
            run_root / "launchers" / "val_anneal" / "submit_grid.py",
            (root / "val_anneal/submit_grid.py").read_bytes(),
        )

        command_rows: list[dict[str, Any]] = []
        for row in rows:
            argv = command_for(root, run_root, row, values)
            command_rows.append({**row, "command": "Command: " + shlex.join(argv)})
        keep_exact(
            run_root / "planned_commands.tsv",
            _write_tsv(PLAN_COLUMNS + ("command",), command_rows),
        )
        verify_source(root, run_root)
        print(
            f"Planned {len(rows)} {benchmark} annealed-{variant} offline-validation runs."
        )
        print(f"Exact commands: {(run_root / 'planned_commands.tsv').as_posix()}")
        if mode == "--dry-run":
            print(
                "Submit with the same deployment overrides: "
                f"bash {wrapper.relative_to(root).as_posix()} --submit"
            )
        return run_root, rows, values
    finally:
        lock.rmdir()


def submit(
    root: Path,
    run_root: Path,
    rows: list[dict[str, Any]],
    values: dict[str, Any],
) -> None:
    if shutil.which("sbatch") is None:
        fail("sbatch is unavailable; run --submit from a Slurm login node")
    (root / "run_logs/slurm").mkdir(parents=True, exist_ok=True)
    verify_source(root, run_root)
    jobs_path = run_root / "jobs.tsv"
    label = f"{values['BENCHMARK']}_annealed_{values['VARIANT']}"
    if jobs_path.exists():
        jobs = _read_tsv(jobs_path)
        prior = [row for row in jobs if row.get("label") == label]
        if prior:
            print(
                f"{label}: already submitted as {prior[-1]['job_id']} "
                f"({prior[-1]['jobs']} tasks)"
            )
            return
    attempt_path = run_root / "submission_attempt.json"
    if attempt_path.exists():
        fail(
            f"A prior submission attempt has no jobs.tsv record; inspect Slurm and "
            f"{attempt_path} before retrying"
        )

    benchmark = str(values["BENCHMARK"])
    variant = str(values["VARIANT"])
    concurrency = int(values["MAX_CONCURRENT"])
    job_name = f"val-{benchmark[:4]}-{'av' if variant == 'vanilla' else 'af'}"
    time_limit = "48:00:00" if benchmark == "robomimic" else "24:00:00"
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--time={time_limit}",
        f"--array={_slurm_array_spec(len(rows), concurrency)}",
        f"--export=ALL,SWEEP_ROOT={run_root.as_posix()},REPO_ROOT={root.as_posix()}",
        (run_root / "launchers/worker.sh").as_posix(),
    ]
    attempt = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "started",
        "command": command,
    }
    try:
        with attempt_path.open("x", encoding="utf-8") as stream:
            json.dump(attempt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        fail(f"Another submission claimed this sweep; inspect {attempt_path}")

    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if result.returncode != 0:
        fail(
            "sbatch failed; submission_attempt.json was retained for audit.\n"
            + result.stderr.strip()
        )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", job_id):
        fail(
            "sbatch returned an unrecognized job ID; inspect Slurm before retrying: "
            + result.stdout.strip()
        )

    if not jobs_path.exists():
        jobs_path.write_text("label\tjob_id\tjobs\tsubmitted_utc\n", encoding="utf-8")
    with jobs_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{label}\t{job_id}\t{len(rows)}\t"
            f"{datetime.now(UTC).isoformat()}\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    attempt["status"] = "recorded"
    attempt["job_id"] = job_id
    temporary = attempt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(attempt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, attempt_path)
    print(f"{label}: submitted as {job_id} ({len(rows)} tasks)")
    print(f"Job manifest: {jobs_path.as_posix()}")


def run_public(root: Path, benchmark: str, variant: str, mode: str) -> None:
    root = root.resolve()
    if benchmark not in BENCHMARKS or variant not in VARIANTS:
        fail("Expected benchmark pusht|libero|robomimic and variant vanilla|foveated")
    wrapper = _wrapper_path(root, benchmark, variant)
    if not wrapper.is_file():
        fail(f"Missing wrapper: {wrapper}")
    if mode in {"--help", "-h"}:
        print(_help(benchmark, variant, wrapper.relative_to(root)))
        return
    if mode not in {"--dry-run", "--submit"}:
        fail("Expected --dry-run, --submit, or --help")
    if not (root / "val_anneal/train_offline.py").is_file():
        fail("Missing offline trainer: val_anneal/train_offline.py")
    run_root, rows, values = plan(root, benchmark, variant, mode, wrapper)
    if mode == "--submit":
        submit(root, run_root, rows, values)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"--verify-source", "--print-row", "--run-row"}:
        action = args.pop(0)
        if action == "--verify-source":
            if len(args) != 2:
                fail("--verify-source requires REPO_ROOT SWEEP_ROOT")
            verify_source(Path(args[0]).resolve(), Path(args[1]).resolve())
            return
        if len(args) != 3 or not re.fullmatch(r"0|[1-9][0-9]*", args[2]):
            fail(f"{action} requires REPO_ROOT SWEEP_ROOT ARRAY_INDEX")
        print_or_run_row(
            Path(args[0]).resolve(),
            Path(args[1]).resolve(),
            int(args[2]),
            execute=action == "--run-row",
        )
        return

    if len(args) not in {3, 4}:
        fail(
            "Usage: submit_grid.py REPO_ROOT pusht|libero|robomimic "
            "vanilla|foveated [--dry-run|--submit|--help]"
        )
    root, benchmark, variant = args[:3]
    mode = args[3] if len(args) == 4 else "--dry-run"
    run_public(Path(root), benchmark, variant, mode)


if __name__ == "__main__":
    main()
