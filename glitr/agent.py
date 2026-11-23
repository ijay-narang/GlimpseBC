"""GLiTR robot-action training without repository policy networks or flow math."""

from dataclasses import asdict

import torch

from glitr.config import GLiTRConfig
from mip.flop_accounting import FlopLedger
from mip.models.encoders.glitr import GLiTRPolicy

CHECKPOINT_FORMAT = "glitr_robot_actions_v2"


def _namespace_to_glitr_config(config):
    if isinstance(config, GLiTRConfig):
        return config
    return GLiTRConfig(
        **{
            key: getattr(config, key)
            for key in GLiTRConfig.__dataclass_fields__
            if hasattr(config, key)
        }
    )


class GLiTRAgent:
    """Adapt the unchanged upstream model to the benchmark runner interface."""

    def __init__(self, config):
        self.config = config
        self.glitr = _namespace_to_glitr_config(config.glitr)
        self.stage = self.glitr.stage
        if config.optimization.loss_type != "regression":
            raise ValueError(
                "GLiTR robot actions require optimization.loss_type=regression"
            )
        if config.network.network_type != "glitr":
            raise ValueError("GLiTR uses its own temporal head; select network=glitr")
        if self.stage not in {"teacher", "student", "eval"}:
            raise ValueError(f"Unknown GLiTR stage: {self.stage}")
        # A full-frame teacher is shared by every crop-specific student.  Keep
        # crop size out of the teacher signature and add it only where it is
        # actually part of the model being trained: the student.
        self.teacher_signature = {
            "env_name": str(config.task.env_name),
            "env_type": str(config.task.env_type),
            "task_id": getattr(config.task, "libero_task_id", None),
            "camera_key": self.glitr.camera_key,
            "image_shape": tuple(
                int(value)
                for value in config.task.shape_meta["obs"][self.glitr.camera_key][
                    "shape"
                ]
            ),
            "backbone": self.glitr.backbone,
            "obs_steps": int(config.task.obs_steps),
            "horizon": int(config.task.horizon),
            "act_dim": int(config.task.act_dim),
        }
        self.student_signature = {
            **self.teacher_signature,
            "glimpse_patches": self.glitr.num_patches_in_glimpse,
        }
        self.signature = (
            self.teacher_signature
            if self.stage == "teacher"
            else self.student_signature
        )
        device = config.optimization.device
        teacher = None
        if self.stage != "teacher":
            if not self.glitr.teacher_checkpoint:
                raise ValueError("GLiTR student/eval requires glitr.teacher_checkpoint")
            state = self._read_checkpoint(
                self.glitr.teacher_checkpoint, expected_stage="teacher"
            )
            # Reconstruct the teacher with the settings it was trained with.
            # Its legacy-required glimpse value is not inherited from the
            # student and has no bearing on the full-frame teacher policy.
            teacher_glitr = GLiTRConfig(**state["glitr"])
            teacher = GLiTRPolicy(config.task, teacher_glitr).to(device)
            teacher.load_state_dict(state["model"])
            teacher.eval().requires_grad_(False)
        self.model = GLiTRPolicy(config.task, self.glitr, teacher=teacher).to(device)
        core = self.model.video_transformer
        # Upstream uses one AdamW per component. Disjoint groups preserve those
        # independent updates while reusing the benchmark runner's scheduling.
        scale = config.optimization.batch_size / 128.0
        head_lr = 1e-4 if self.stage == "teacher" else 1e-5
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": core.backbone.parameters(),
                    "lr": self.glitr.base_lr_backbone * scale,
                },
                {
                    "params": core.temporal_head.parameters(),
                    "lr": (self.glitr.base_lr_temphead or head_lr) * scale,
                },
                {
                    "params": core.temploc_head.parameters(),
                    "lr": (self.glitr.base_lr_temploc or head_lr) * scale,
                },
            ],
            weight_decay=self.glitr.weight_decay,
        )
        self.flops = FlopLedger(config)

    def update(self, act, obs, delta_t=None):
        del delta_t  # Runner compatibility; GLiTR has no flow-time input.
        return self.flops.run(
            "train",
            lambda: self._update_impl(act, obs),
            inputs=(act, obs[self.glitr.camera_key]),
            batch_size=len(act),
            metadata=(self.stage,),
            compiled=False,
        )

    def _update_impl(self, act, obs):
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(obs, act)
        loss = (
            output.action_loss
            + self.glitr.spatial_loss_weight * output.spatial_loss
            + self.glitr.temporal_loss_weight * output.temporal_loss
        )
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.detach(),
            "action_loss": output.action_loss.detach(),
            "spatial_loss": output.spatial_loss.detach(),
            "temporal_loss": output.temporal_loss.detach(),
        }

    def sample(self, act_0, obs, num_steps=-1, use_ema=True):
        # The runner supplies these to every policy. GLiTR has no noise input,
        # iterative sampler, or exponential-moving-average model.
        del num_steps, use_ema
        expected = (
            len(act_0),
            self.teacher_signature["horizon"],
            self.teacher_signature["act_dim"],
        )
        if tuple(act_0.shape) != expected:
            raise ValueError(f"GLiTR action buffer must have shape {expected}")
        return self.flops.run(
            "eval",
            lambda: self._sample_impl(obs),
            inputs=(obs[self.glitr.camera_key],),
            batch_size=len(act_0),
            metadata=(self.stage, tuple(m.training for m in self.model.modules())),
            compiled=False,
        )

    @torch.no_grad()
    def _sample_impl(self, obs):
        return self.model(obs).actions

    def save(self, path, training_state=None):
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "stage": "student" if self.stage == "eval" else self.stage,
                "signature": self.signature,
                "glitr": asdict(self.glitr),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_state": training_state,
            },
            path,
        )

    def _read_checkpoint(self, path, expected_stage=None):
        state = torch.load(
            path, map_location=self.config.optimization.device, weights_only=False
        )
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                "Incompatible GLiTR checkpoint: old flow-policy hybrids and original "
                "video classifiers cannot load as robot-action policies. Retrain the teacher."
            )
        checkpoint_stage = state.get("stage")
        if expected_stage is not None and checkpoint_stage != expected_stage:
            raise ValueError(f"Expected a GLiTR {expected_stage} checkpoint")
        expected_signature = (
            self.teacher_signature
            if checkpoint_stage == "teacher"
            else self.student_signature
        )
        if state.get("signature") != expected_signature:
            raise ValueError(
                "GLiTR checkpoint camera, model, observation or action shapes differ"
            )
        return state

    def load(self, path, load_optimizer=False):
        stage = "student" if self.stage == "eval" else self.stage
        state = self._read_checkpoint(path, expected_stage=stage)
        self.model.load_state_dict(state["model"])
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
        return state.get("training_state")

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
