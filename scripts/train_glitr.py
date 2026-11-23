"""Launch one GLiTR teacher or student using the existing robotics adapter."""

import argparse
import os
from pathlib import Path

from _launch import (
    ROOT,
    add_common_arguments,
    main,
    make_launch,
    quote_path,
    task_overrides,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_arguments(result)
    result.add_argument("--stage", required=True, choices=("teacher", "student"))
    result.add_argument(
        "--crop-size",
        type=int,
        choices=(32, 48, 64),
        help="student only; pixels after GLiTR's internal 224px resize",
    )
    result.add_argument("--teacher-checkpoint", type=Path, help="required for student")
    result.add_argument(
        "--pretrained-dir",
        type=Path,
        help="iBOT checkpoint directory; defaults to GLITR_PRETRAINED_DIR or glitr/pretrained",
    )
    return result


def build(args):
    task_args, task_label = task_overrides(args)
    if args.stage == "teacher":
        if args.crop_size is not None or args.teacher_checkpoint is not None:
            raise ValueError(
                "--crop-size and --teacher-checkpoint apply only to student"
            )
    elif args.crop_size is None or args.teacher_checkpoint is None:
        raise ValueError("student requires --crop-size and --teacher-checkpoint")
    camera = {
        "robomimic": "agentview_image",
        "pusht": "image",
        "libero": "agentview_rgb",
    }[args.benchmark]
    pretrained = (
        (
            args.pretrained_dir
            or Path(
                os.environ.get(
                    "GLITR_PRETRAINED_DIR", ROOT / "glitr/pretrained"
                )
            )
        )
        .expanduser()
        .resolve()
    )
    model_args = [
        "network=glitr",
        "optimization.loss_type=regression",
        "++optimization.eval_num_steps=[1]",
        f"glitr.stage={args.stage}",
        f"glitr.camera_key={camera}",
        "glitr.input_size=224",
        f"glitr.num_patches_in_glimpse={args.crop_size // 16 if args.crop_size else 3}",
        f"glitr.pretrained_dir={quote_path(pretrained)}",
    ]
    geometry = "full" if args.stage == "teacher" else f"crop{args.crop_size}"
    run_id = (
        f"{args.benchmark}_{task_label}_glitr_{args.stage}_{geometry}_seed0_eval10000"
    )
    launch = make_launch(args, task_args, run_id, model_args, glitr=True)
    launch.required_files = (
        ROOT / "third_party/GliTr/model.py",
        pretrained / "ibot_vits_16_checkpoint_teacher.pth",
    )
    if args.stage == "teacher":
        launch.teacher_export = launch.output_dir / "teacher.pt"
        launch.command.append(
            f"glitr.export_checkpoint={quote_path(launch.teacher_export)}"
        )
    else:
        teacher = args.teacher_checkpoint.expanduser().resolve()
        launch.required_files += (teacher,)
        launch.command.append(f"glitr.teacher_checkpoint={quote_path(teacher)}")
    return launch


if __name__ == "__main__":
    raise SystemExit(main(parser(), build))
