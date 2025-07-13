
from vri.policies import base_policy as _base_policy
import numpy as np
from typing import Dict
from typing_extensions import override
import tree

class ActionChunkPolicy(_base_policy.BasePolicy):
    def __init__(self, policy: _base_policy.BasePolicy, action_chunk_size: int):
        """
        Initialize the ActionChunkPolicy with the given policy.

        Args:
            policy (BasePolicy): The base policy to be used.
            action_chunk_size (int): The size of the action chunk.
        """
        
        self._policy = policy
        self._action_chunk_size = action_chunk_size
        self._cur_step = 0
        self._last_results:Dict[str, np.ndarray] | None = None
    
    @override
    def infer(self, obs:Dict) -> Dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0
        
        results = tree.map_structure(
            lambda x: x[self._cur_step, ...],
            self._last_results
        )
        self._cur_step += 1

        if self._cur_step >= self._action_chunk_size:
            self._last_results = None
        
        return results
    
    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0