import abc

class Agent(abc.ABC):

    @abc.abstractmethod
    def get_action(self, observation: dict) -> dict:
        """
        Get the action to be performed based on the current observation.
        
        Args:
            observation (dict): The current observation from the environment.
        
        Returns:
            dict: The action to be performed.
        """
    
    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        """