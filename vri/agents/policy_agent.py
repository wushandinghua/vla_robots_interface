from typing_extensions import override
from vri.policies import base_policy as _base_policy
from vri.agents import base_agent as _base_agent

class PolicyAgent(_base_agent.Agent):
    """
    A policy agent that uses a specific policy to determine actions based on observations.
    """

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        """
        Initialize the PolicyAgent with a given policy.

        Args:
            policy (BasePolicy): The policy to use for action selection.
        """
        self._policy = policy

    @override
    def get_action(self, observation: dict) -> dict:
        """
        Get the action to be performed based on the current observation.

        Args:
            observation (dict): The current observation from the environment.

        Returns:
            dict: The action to be performed.
        """
        return self._policy.infer(observation)

    @override
    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        """
        self._policy.reset()