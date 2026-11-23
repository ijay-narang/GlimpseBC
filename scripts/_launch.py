"""Shared command construction for the public single-run launchers.

Only the standard library is needed here. Training runs in the Python environment
used to invoke the launcher, with the existing benchmark trainers and Hydra configs.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SIZES = {"robomimic": 84, "pusht": 96, "libero": 128}
CROP_SIZES = {
    "robomimic": (32, 48, 64),
    "pusht": (24, 32, 48, 64),
    "libero": (32, 48, 64, 96),
}
ROBOMIMIC_TASKS = ("can", "lift", "square", "tool_hang")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_common_arguments(parser):
    parser.add_argument("--benchmark", required=True, choices=tuple(SOURCE_SIZES))
    parser.add_argument(
        "--task",
        help="Robomimic: can/lift/square/tool_hang; LIBERO-Spatial: 0-9; omit for PushT",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="model device: cpu or cuda (default: cuda)",
    )
    parser.add_argument("--steps", type=positive_int, default=200000)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new run directory (default: run_logs/<experiment>/<UTC timestamp>)",
    )


def task_overrides(args):
    if args.benchmark == "robomimic":
        if args.task not in ROBOMIMIC_TASKS:
            raise ValueError(
                "--task must be can, lift, square, or tool_hang for Robomimic"
            )
        return [f"task={args.task}_ph_image", "+task.image_shape=[84,84]"], args.task
    if args.benchmark == "pusht":
        if args.task is not None:
            raise ValueError("omit --task for PushT")
        return ["task=pusht_image", "task.num_envs=20"], "pusht"
    if args.task not in tuple(str(i) for i in range(10)):
        raise ValueError("--task must be a LIBERO-Spatial task ID from 0 to 9")
    data_root = Path(os.environ.get("LIBERO_DATA_ROOT", ROOT / "libero_datasets"))
    return [
        "task=libero_spatial_0_image",
        f"task.libero_task_id={args.task}",
        f"task.dataset_root={quote_path(data_root)}",
        "task.num_envs=1",
    ], f"libero_spatial_task{args.task}"


def quote_path(path):
    """Quote for Hydra, not for a shell; subprocess receives an argument list."""
    return json.dumps(Path(path).expanduser().resolve().as_posix())


def run_environment(benchmark, device="cuda"):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("HF_HOME", str(ROOT / ".cache/hf_cache"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(env["HF_HOME"]) / "hub"))
    env.setdefault("TMPDIR", str(ROOT / ".cache/tmp"))
    env["MIP_MEASURE_FLOPS"] = "1"
    env["MIP_DISABLE_GLOBAL_CHECKPOINTS"] = "1"
    if benchmark != "pusht":
        env.setdefault("MUJOCO_GL", "osmesa" if device == "cpu" else "egl")
        env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
        if env["MUJOCO_GL"] == "egl":
            env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    if benchmark == "libero":
        env.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".cache/libero_config"))
        env.setdefault("LIBERO_DATA_ROOT", str(ROOT / "libero_datasets"))
    return env


@dataclass
class Launch:
    command: list[str]
    output_dir: Path
    environment: dict[str, str]
    required_files: tuple[Path, ...] = ()
    teacher_export: Path | None = None


def make_launch(args, task_args, run_id, model_args, *, glitr=False):
    # LIBERO's isolated environment uses Python 3.10 (before datetime.UTC).
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # noqa: UP017
    output = (
        (args.output_dir or ROOT / "run_logs" / run_id / stamp).expanduser().resolve()
    )
    entrypoint = f"examples/train_{args.benchmark}.py"
    config_name = "glitr" if glitr else "main"
    if glitr and args.benchmark == "robomimic":
        entrypoint = "glitr/train_robomimic.py"
        config_name = "glitr_robomimic"
    common = [
        f"optimization.device={args.device}",
        "optimization.seed=0",
        "optimization.eval_seed=10000",
        "optimization.deterministic=false",
        f"optimization.batch_size={args.batch_size}",
        f"optimization.gradient_steps={args.steps}",
        "optimization.use_compile=false",
        "optimization.use_cudagraphs=false",
        "optimization.auto_resume=false",
        f"log.eval_freq={args.steps}",
        f"log.eval_episodes={50 if args.benchmark == 'libero' else 100}",
        "log.save_freq=20000",
        "log.log_freq=1000",
        f"log.log_dir={quote_path(output)}",
        f"log.exp_name={run_id}",
        f"log.group={args.benchmark}_{'glitr' if glitr else args.method}",
    ]
    return Launch(
        [
            sys.executable,
            str(ROOT / entrypoint),
            "--config-name",
            config_name,
            *task_args,
            *model_args,
            *common,
        ],
        output,
        run_environment(args.benchmark, args.device),
    )


def execute(launch):
    for path in launch.required_files:
        if not path.is_file():
            raise ValueError(f"Missing required file: {path}. See README.md for setup.")
    if launch.output_dir.exists():
        raise ValueError(
            f"Output directory already exists: {launch.output_dir}; choose a new --output-dir"
        )
    launch.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TMPDIR"):
        Path(launch.environment[key]).mkdir(parents=True, exist_ok=True)
    print(
        f"Training seed: 0; evaluation seed: 10000\nOutput: {launch.output_dir}",
        flush=True,
    )
    if launch.teacher_export:
        print(f"Teacher will be exported to: {launch.teacher_export}", flush=True)
    print("Command: " + shlex.join(launch.command), flush=True)
    # No shell, preview stage, or confirmation: launch the trainer immediately.
    return subprocess.run(launch.command, cwd=ROOT, env=launch.environment).returncode


def main(parser, build, argv=None):
    args = parser.parse_args(argv)
    try:
        return execute(build(args))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        return 130
