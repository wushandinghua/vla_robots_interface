import abc

class Subscriber(abc.ABC):
    """
    used to visualize the environment and agent
    """

    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """
        Called at the start of each episode.
        """
    
    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """
        Called at the end of each episode.
        """
    
    @abc.abstractmethod
    def on_step(self, observation: dict, action: dict) -> None:
        """
        Called at each step of the episode.
        
        Args:
            observation (dict): The current observation from the environment.
            action (dict): The action taken by the agent.
        """