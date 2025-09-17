import collections
import time
import copy
from typing import Optional, List
import dm_env
import numpy as np

from vri.utils import constants
from vri.robot_devices import agilex_piper


class RealEnv:
    """
    Environment for agilex piper
    Action space:      [left_arm_qpos (6),             # joint position
                        left_gripper_positions (1),    # gripper position
                        right_arm_qpos (6),            
                        right_gripper_positions (1)]

    Observation space: {"state": [left_arm_qpos (6),             # joint position
                            left_gripper_positions (1),    # gripper position
                            right_arm_qpos (6),            
                            right_gripper_positions (1)]
                                        
                        "images": {"cam1": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam2": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam3": (480x640x3),         # h, w, c, dtype='uint8'
                        }
    """

    def __init__(self, reset_position: Optional[List[float]] = None, reset_type: int = 3):
        self._reset_position = reset_position
        self._reset_type = reset_type

        # new agilex piper controller
        self.robot = agilex_piper.PiperPlay(reset_type=self._reset_type, reset_position=self._reset_position)
        self._chunk_size = constants.CHUNK_SIZE
        self._last_observation = None

        # if setup_robots:
        #     self.setup_robots()

    def setup_robots(self):
        # reboot robot
        command = "reboot robot"
        # self.robot.back_home()
        print(f"real env setup cmd:{command}, sleep time:{constants.DT}")

    def get_observation(self):
        ret = self.robot.capture_observation()
        self._last_observation = ret
        return ret
        

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        if not fake:
            # Reboot robot 
            self.setup_robots()
            print("real env setup finished")
        # return dm_env.TimeStep(
        #     step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        # )
        return self.get_observation()

    def step(self, action):
        cur_step = action.get("cur_step")
        action = action["actions"]
        assert action.shape[-1] == 14
        self.robot.send_action(action)
        # return dm_env.TimeStep(
        #     step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        # )
        if cur_step >= (self._chunk_size - 1):
            time.sleep(constants.DT)
            return self.get_observation()
        
        # return self._last_observation
        return self.get_observation()


def make_real_env(reset_position: Optional[List[float]] = None, reset_type: int = 3) -> RealEnv:
    return RealEnv(reset_position=reset_position, reset_type=reset_type)