from typing import List, Optional  # noqa: UP035

import einops
from vri.utils import image_tools
from typing_extensions import override
import numpy as np
import collections
from vri.environments import base_env as _base_env
from vri.environments import piper_real_env as _real_env
from vri.utils.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST
import time

class PiperEnvironment(_base_env.Environment):
    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        instruction: str = None,
        reset_type: int = 3 # 0: no reset, 1: reset when robot connect, 2: reset when robot disconnect, 3: reset when robot connect and disconnect
    ) -> None:
        self._env = _real_env.make_real_env(reset_position=reset_position, reset_type=reset_type)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None
        self.instruction = instruction
        # action queue for the last 10 actions
        self._exec_action_queue = collections.deque(maxlen=20)
        # cache images of cam high
        self._cam_high_images = []
        self._cam_left_wrist_images = []
        self._cam_right_wrist_images = []

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        """Check if the current episode is complete."""
        end = time.time()
        if len(self._exec_action_queue) < 20:
            return False
        arr = np.array(self._exec_action_queue) # shape=(10, 14)
        diff = arr.max(axis=0) - arr.min(axis=0)
        return np.all(diff < 5)

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts
        
        self._cam_high_images.append(obs[f"observation.images.{CAM_HIGH}"])
        self._cam_left_wrist_images.append(obs[f"observation.images.{CAM_LEFT_WRIST}"])
        self._cam_right_wrist_images.append(obs[f"observation.images.{CAM_RIGHT_WRIST}"])

        for key in obs.keys():
            if "images" not in key: continue
            if obs[key].shape[0] == self._render_height and obs[key].shape[1] == self._render_width: continue
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs[key], self._render_height, self._render_width)
            )
            # obs[key] = einops.rearrange(img, "h w c -> c h w")
            obs[key] = img

        return {
            CAM_HIGH: obs[f"observation.images.{CAM_HIGH}"],
            CAM_LEFT_WRIST: obs[f"observation.images.{CAM_LEFT_WRIST}"],
            CAM_RIGHT_WRIST: obs[f"observation.images.{CAM_RIGHT_WRIST}"],
            "state": np.array(obs["observation.state"]),
            "prompt": self.instruction,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._exec_action_queue.append(action["actions"])
        self._ts = self._env.step(action)

    @property
    def gen_fake_observation(self):
        fake_observation = {
            CAM_HIGH: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            CAM_LEFT_WRIST: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            CAM_RIGHT_WRIST: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "state": np.random.rand(14),
            "prompt": "do something",
        }
        return fake_observation
