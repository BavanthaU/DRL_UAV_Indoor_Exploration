from __future__ import annotations

from typing import Iterable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class VecDictFrameStack(VecEnvWrapper):
    """Stack the last ``n`` observations for each specified key of a dict observation."""

    def __init__(
        self,
        venv: VecEnv,
        n_stack: int = 4,
        stack_keys: Iterable[str] | None = None,
    ) -> None:
        super().__init__(venv)
        if n_stack < 1:
            raise ValueError("n_stack must be >= 1.")

        if not isinstance(venv.observation_space, gym.spaces.Dict):
            raise TypeError("VecDictFrameStack requires a Dict observation space.")

        self.n_stack = int(n_stack)
        self.stack_keys = set(stack_keys) if stack_keys is not None else set(venv.observation_space.spaces.keys())
        invalid_keys = self.stack_keys.difference(venv.observation_space.spaces.keys())
        if invalid_keys:
            raise KeyError(f"Unknown observation keys requested for stacking: {sorted(invalid_keys)}")

        self.key_shapes: dict[str, tuple[int, ...]] = {}
        self.buffers: dict[str, np.ndarray] = {}

        new_spaces: dict[str, gym.Space] = {}
        for key, space in venv.observation_space.spaces.items():
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(f"Unsupported observation space type for key '{key}': {type(space)}")

            self.key_shapes[key] = space.shape
            if key in self.stack_keys:
                if len(space.shape) == 0:
                    raise ValueError(f"Cannot stack scalar observation for key '{key}'.")
                stacked_shape = (space.shape[0] * self.n_stack,) + space.shape[1:]
                low = np.repeat(space.low, self.n_stack, axis=0)
                high = np.repeat(space.high, self.n_stack, axis=0)
                new_spaces[key] = gym.spaces.Box(low=low, high=high, dtype=space.dtype)
                buffer_shape = (self.num_envs, self.n_stack) + space.shape
                self.buffers[key] = np.zeros(buffer_shape, dtype=space.dtype)
            else:
                new_spaces[key] = space

        self.observation_space = gym.spaces.Dict(new_spaces)

    def reset(self) -> dict[str, np.ndarray]:
        obs = self.venv.reset()
        return self._stack_observations(obs, reset=True)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        stacked_obs = self._stack_observations(obs, reset=False, dones=dones)
        return stacked_obs, rewards, dones, infos

    def _stack_observations(
        self,
        obs: dict[str, np.ndarray],
        reset: bool,
        dones: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        stacked: dict[str, np.ndarray] = {}
        for key, value in obs.items():
            if key not in self.buffers:
                stacked[key] = value
                continue

            buffer = self.buffers[key]
            if reset:
                buffer[:] = value[:, None, ...]
            else:
                buffer[:, :-1, ...] = buffer[:, 1:, ...]
                buffer[:, -1, ...] = value
                if dones is not None and np.any(dones):
                    done_idx = np.where(dones)[0]
                    buffer[done_idx, :, ...] = value[done_idx][:, None, ...]

            stacked[key] = self._reshape_buffer(buffer, key)
        return stacked

    def _reshape_buffer(self, buffer: np.ndarray, key: str) -> np.ndarray:
        orig_shape = self.key_shapes[key]
        if len(orig_shape) == 1:
            return buffer.reshape(self.num_envs, self.n_stack * orig_shape[0])
        new_shape = (self.num_envs, self.n_stack * orig_shape[0]) + orig_shape[1:]
        return buffer.reshape(new_shape)
