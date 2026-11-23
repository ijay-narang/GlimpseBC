"""Gymnasium adapter for official single-task LIBERO evaluation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mip.data.libero_utils import (
    get_libero_task_id,
    load_libero_init_states,
    make_libero_suite,
)
from mip.env_utils import MultiStepWrapper, VideoRecorder, VideoRecordingWrapper

DEFAULT_OBS_KEY_MAPPING = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
    "ee_states": "robot0_eef_pos",
}


class LiberoEnv(gym.Env):
    """Expose one official LIBERO task through the Gymnasium API."""

    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(
        self,
        task_config: Any,
        env_index: int = 0,
        seed: int | None = None,
        *,
        _suite=None,
        _env_factory: Callable[..., Any] | None = None,
    ):
        super().__init__()
        if str(task_config.obs_type) != "image":
            raise ValueError(
                "The modular LIBERO pipeline currently supports image observations"
            )
        self.task_config = task_config
        self.env_index = int(env_index)
        self._seed = seed
        self._reset_count = 0
        self._render_cache: np.ndarray | None = None
        uses_official_suite = _suite is None
        self._suite = _suite if _suite is not None else make_libero_suite(task_config)
        self.task_id = get_libero_task_id(task_config)
        self.task = self._suite.get_task(self.task_id)
        if uses_official_suite:
            init_states = load_libero_init_states(self.task)
        else:
            init_states = self._suite.get_task_init_states(self.task_id)
        self.init_states = self._to_numpy(init_states)
        if len(self.init_states) == 0:
            raise ValueError("LIBERO benchmark returned no fixed initial states")

        shape_meta = task_config.shape_meta
        self.rgb_keys = []
        self.lowdim_keys = []
        self.observation_space = spaces.Dict()
        for key, attr in shape_meta["obs"].items():
            key = str(key)
            shape = tuple(int(dim) for dim in attr["shape"])
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
                self.observation_space[key] = spaces.Box(
                    low=0.0, high=1.0, shape=shape, dtype=np.float32
                )
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)
                self.observation_space[key] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
                )
            else:
                raise ValueError(f"Unsupported LIBERO observation type {obs_type!r}")
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(task_config.act_dim),),
            dtype=np.float32,
        )

        configured_mapping = getattr(task_config, "libero_obs_key_mapping", None)
        self.obs_key_mapping = dict(DEFAULT_OBS_KEY_MAPPING)
        if configured_mapping:
            self.obs_key_mapping.update(dict(configured_mapping))

        if _env_factory is None:
            try:
                from libero.libero import get_libero_path
                from libero.libero.envs import OffScreenRenderEnv
            except ImportError as exc:
                raise ImportError(
                    "Official LIBERO is required for environment rollouts. Run "
                    "`bash scripts/libero/utils/setup_libero_env.sh`."
                ) from exc
            bddl_path = self._suite.get_task_bddl_file_path(self.task_id)
            if not bddl_path:
                bddl_path = str(
                    Path(get_libero_path("bddl_files"))
                    / self.task.problem_folder
                    / self.task.bddl_file
                )
            _env_factory = OffScreenRenderEnv
        else:
            bddl_path = getattr(self.task, "bddl_file", "mock.bddl")

        camera_shapes = [
            tuple(int(dim) for dim in shape_meta["obs"][key]["shape"])
            for key in self.rgb_keys
        ]
        if not camera_shapes:
            raise ValueError("LIBERO environment requires at least one RGB observation")
        self.render_obs_key = str(
            getattr(task_config, "render_obs_key", self.rgb_keys[0])
        )
        if self.render_obs_key not in self.rgb_keys:
            raise ValueError(
                f"render_obs_key={self.render_obs_key!r} is not an RGB observation"
            )
        heights = {shape[-2] for shape in camera_shapes}
        widths = {shape[-1] for shape in camera_shapes}
        if len(heights) != 1 or len(widths) != 1:
            raise ValueError(
                "Official LIBERO requires all configured cameras to share a size"
            )
        camera_names = [
            self.obs_key_mapping[key].removesuffix("_image") for key in self.rgb_keys
        ]
        self.camera_names = camera_names
        self.env = _env_factory(
            bddl_file_name=str(bddl_path),
            camera_names=camera_names,
            camera_heights=heights.pop(),
            camera_widths=widths.pop(),
            horizon=int(task_config.max_episode_steps),
            # LIBERO replaces Robosuite's horizon `done` with task success while
            # Robosuite still marks itself terminated internally. Keep the
            # simulator horizon open and let MultiStepWrapper own truncation.
            ignore_done=True,
            hard_reset=bool(getattr(task_config, "libero_hard_reset", True)),
        )
        self.settle_steps = int(getattr(task_config, "libero_settle_steps", 5))
        if seed is not None:
            self.seed(seed)

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def seed(self, seed: int | None = None):
        self._seed = seed
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(int(seed))
        return [seed]

    def _initial_state_index(self) -> int:
        if self._seed is not None:
            value = int(self._seed)
        else:
            value = self._reset_count + self.env_index
        return value % len(self.init_states)

    def _process_observation(self, raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        for key in self.rgb_keys + self.lowdim_keys:
            source_key = self.obs_key_mapping.get(key, key)
            if source_key not in raw_obs:
                available = ", ".join(sorted(raw_obs))
                raise KeyError(
                    f"LIBERO observation {source_key!r} (for {key!r}) is missing; "
                    f"available: {available}"
                )
            value = np.asarray(raw_obs[source_key])
            if key in self.rgb_keys:
                if key == self.render_obs_key:
                    self._render_cache = value
                expected_channels = self.observation_space[key].shape[0]
                if value.shape[-1] == expected_channels:
                    value = np.moveaxis(value, -1, -3)
                elif value.shape[-3] != expected_channels:
                    raise ValueError(
                        f"LIBERO camera {source_key!r} has unsupported shape {value.shape}"
                    )
                value = value.astype(np.float32)
                if value.max() > 1.0:
                    value /= 255.0
            else:
                value = value.astype(np.float32)
            expected_shape = self.observation_space[key].shape
            if value.shape != expected_shape:
                raise ValueError(
                    f"LIBERO observation {key!r} has shape {value.shape}; "
                    f"expected {expected_shape}"
                )
            obs[key] = np.ascontiguousarray(value)
        return obs

    def reset(self, *, seed: int | None = None, options=None):
        del options
        if seed is not None:
            self.seed(seed)
        self.env.reset()
        init_state_id = self._initial_state_index()
        raw_obs = self.env.set_init_state(self.init_states[init_state_id])
        for _ in range(self.settle_steps):
            raw_obs, _, done, _ = self.env.step(
                np.zeros(self.action_space.shape, dtype=np.float32)
            )
            if done:
                break
        self._seed = None
        self._reset_count += 1
        return self._process_observation(raw_obs), {
            "libero_task": self.task.name,
            "libero_language": self.task.language,
            "libero_init_state_id": init_state_id,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"LIBERO action has shape {action.shape}; expected {self.action_space.shape}"
            )
        raw_obs, reward, done, info = self.env.step(action)
        success = bool(self.env.check_success())
        reward = max(float(reward), float(success))
        info = dict(info or {})
        info["success"] = success
        terminated = success
        truncated = bool(done) and not success
        return self._process_observation(raw_obs), reward, terminated, truncated, info

    def render(self):
        if self._render_cache is None:
            raise RuntimeError("Call reset or step before rendering LIBERO")
        image = np.asarray(self._render_cache)
        if image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
            image = np.moveaxis(image, 0, -1)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image[::-1])

    def close(self):
        if hasattr(self, "env"):
            self.env.close()


def make_libero_env(task_config: Any, idx: int, render: bool = False, seed=None):
    """Return a constructor suitable for ``gym.vector.SyncVectorEnv``."""

    def thunk():
        env = LiberoEnv(task_config, env_index=idx, seed=seed)
        video_recorder = VideoRecorder.create_h264(
            fps=10,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            file_path="results/libero.mp4" if render else None,
            steps_per_render=1,
        )
        env = MultiStepWrapper(
            env,
            n_obs_steps=int(task_config.obs_steps),
            n_action_steps=int(task_config.act_steps),
            max_episode_steps=int(task_config.max_episode_steps),
            reward_agg_method="max",
        )
        if seed is not None:
            env.seed(int(seed) + idx)
        return env

    return thunk


def make_vec_env(task_config: Any, seed=None):
    """Create synchronous vectorized LIBERO environments."""
    return gym.vector.SyncVectorEnv(
        [
            make_libero_env(task_config, idx, False, seed=seed)
            for idx in range(int(task_config.num_envs))
        ]
    )
