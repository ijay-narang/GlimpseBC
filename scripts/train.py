"""Launch one GlimpseBC or baseline training run using existing trainers."""

import argparse

from _launch import (
    CROP_SIZES,
    SOURCE_SIZES,
    add_common_arguments,
    main,
    make_launch,
    positive_int,
    task_overrides,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_arguments(result)
    result.add_argument(
        "--method",
        required=True,
        choices=("full", "center", "glimpse", "annealed", "foveated"),
    )
    result.add_argument("--policy", required=True, choices=("rnn", "chiunet"))
    result.add_argument(
        "--crop-size",
        type=positive_int,
        help="32/48/64; also PushT 24 or LIBERO 96; omit for full",
    )
    result.add_argument(
        "--start-size",
        type=positive_int,
        help="starting source width in pixels; required for annealed",
    )
    result.add_argument(
        "--anneal-steps",
        type=positive_int,
        help="annealed only; default 100000 optimizer updates",
    )
    return result


def build(args):
    task_args, task_label = task_overrides(args)
    if args.method == "full":
        if args.crop_size is not None:
            raise ValueError(
                "omit --crop-size for full; benchmark baseline preprocessing is retained"
            )
    elif args.crop_size not in CROP_SIZES[args.benchmark]:
        raise ValueError(
            f"--crop-size for {args.benchmark} must be one of {CROP_SIZES[args.benchmark]}"
        )
    if args.method != "annealed" and (
        args.start_size is not None or args.anneal_steps is not None
    ):
        raise ValueError(
            "--start-size and --anneal-steps apply only to --method annealed"
        )
    if args.method == "foveated" and args.benchmark == "robomimic":
        raise ValueError("foveated is supported here only for PushT and LIBERO")

    loss = "regression" if args.policy == "rnn" else "flow"
    encoder = {
        "full": "image",
        "center": "image",
        "glimpse": "vanilla_saccade_image",
        "annealed": "annealed_vanilla_saccade_image",
        "foveated": "foveated_saccade_image",
    }[args.method]
    model_args = [
        f"network={args.policy}",
        f"network.encoder_type={encoder}",
        f"optimization.loss_type={loss}",
    ]
    if args.benchmark == "libero" and args.policy == "chiunet":
        model_args.append("task.horizon=16")
    if args.benchmark != "pusht":
        model_args.append(
            "++optimization.eval_num_steps="
            + ("[1]" if args.policy == "rnn" else "[9,3,1]")
        )
    geometry = "full"
    if args.method == "full":
        # PushT baseline uses the established random 84px crop from a 96px source.
        if args.benchmark != "pusht":
            model_args.append("task.crop_shape=null")
    else:
        model_args.append(f"task.crop_shape=[{args.crop_size},{args.crop_size}]")
        geometry = f"crop{args.crop_size}"
    if args.method == "center":
        model_args.append("task.random_crop=false")
    elif args.method == "annealed":
        source = SOURCE_SIZES[args.benchmark]
        if args.start_size is None or not args.crop_size <= args.start_size <= source:
            raise ValueError(
                f"--start-size must be between crop size {args.crop_size} and source width {source}"
            )
        duration = args.anneal_steps or 100000
        if duration > args.steps:
            raise ValueError(
                "--anneal-steps cannot exceed --steps; the final crop must be reached during training"
            )
        model_args += [
            f"network.annealed_vanilla_patch_size={args.crop_size}",
            f"network.annealed_vanilla_start_scale={args.start_size / source}",
            f"network.annealed_vanilla_anneal_steps={duration}",
        ]
        geometry += f"_start{args.start_size}_anneal{duration}"
    elif args.method == "foveated":
        model_args += [
            f"network.foveated_patch_size={args.crop_size}",
            f"network.foveated_radius={args.crop_size / 2}",
            "network.foveated_fovea_ratio=0.4",
            "network.foveated_blur_sigma=2.0",
        ]
    run_id = f"{args.benchmark}_{task_label}_{args.method}_{args.policy}_{geometry}_seed0_eval10000"
    return make_launch(args, task_args, run_id, model_args)


if __name__ == "__main__":
    raise SystemExit(main(parser(), build))
