import abc

class Environment(abc.ABC):

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the environment to its initial state.
        """
    
    @abc.abstractmethod
    def is_episode_complete(self) -> bool:
        """
        Check if the current episode has ended.
        
        Returns:
            bool: True if the episode has ended, False otherwise.
        """
    
    @abc.abstractmethod
    def get_observation(self) -> dict:
        """
        Get the current observation from the environment.
        
        Returns:
            dict: The current observation.
        """
    
    @abc.abstractmethod
    def apply_action(self, action: dict) -> None:
        """
        Apply the given action to the environment.
        
        Args:
            action (dict): The action to apply.
        """