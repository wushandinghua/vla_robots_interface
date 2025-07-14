from typing import List, Optional  # noqa: UP035

import einops
from vri.utils import image_tools
from typing_extensions import override
import numpy as np
import collections
from vri.environments import base_env as _base_env
from vri.environments import airbot_real_env as _real_env
from vri.utils.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST

class AirbotEnvironment(_base_env.Environment):
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

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        """Check if the current episode is complete."""
        if len(self._exec_action_queue) < 20:
            return False
        arr = np.array(self._exec_action_queue) # shape=(10, 14)
        diff = arr.max(axis=0) - arr.min(axis=0)
        return np.all(diff < 0.01)

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        """cache images of cam high"""
        self._cam_high_images.append(obs[f"observation.images.{CAM_HIGH}"])
        # for k in list(obs["images"].keys()):
        #     if "_depth" in k:
        #         del obs["images"][k]

        #print("cam wrist image shape before:", np.array(obs["images"][f"{CAM_WRIST}"]).shape)
        #print("cam exterior image shape before:", np.array(obs["images"][f"{CAM_WRIST}"]).shape)
        # for cam_name in obs["images"]:
        #     #img = image_tools.convert_to_uint8(
        #     #    image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
        #     #)
        #     #obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
        #     img = image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
        #     obs["images"][cam_name] = img 
        for key in obs.keys():
            #img = image_tools.convert_to_uint8(
            #    image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            #)
            #obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
            if "images" not in key: continue
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs[key], self._render_height, self._render_width)
            )
            obs[key] = einops.rearrange(img, "h w c -> c h w")

        #print("cam wrist image shape after:", obs["images"][f"{CAM_WRIST}"].shape)
        #print("cam exterior image shape after:", obs["images"][f"{CAM_WRIST}"].shape)
        # return {
        #     "state": obs["qpos"],
        #     "images": obs["images"],
        # }
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
        self._ts = self._env.step(action["actions"])

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