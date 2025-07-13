import logging
import time
from vri.environments import base_env as _environment
from vri.agents import base_agent as _agent
from vri.subscribers import subscriber as _subscriber


class Runtime:

    def __init__(
        self, 
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscriber: _subscriber.Subscriber,
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0
    ) -> None:
        """
        Initialize the Runtime with the given environment and agent.

        Args:
            environment (Environment): The environment to run.
            agent (Agent): The agent to control the environment.
            subscriber (Subscriber): Subscriber to visualize video.
            max_hz (float): Maximum frequency of actions.
            num_episodes (int): Number of episodes to run.
            max_episode_steps (int): Maximum steps per episode.
        """
        self._environment = environment
        self._agent = agent
        self._subscriber = subscriber
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0
    
    def run(self) -> int:
        for _ in range(self._num_episodes):
            self._run_episode()
        
        self._environment.reset()
        action_status_type = -1
        if not self._in_episode and self._environment.is_episode_complete():
            logging.info("episode completed successfully.")
            action_status_type = 0
        elif not self._in_episode and self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps:
            logging.info("episode completed due to max steps reached.")
            action_status_type = 1
        return action_status_type
    
    def _run_episode(self) -> None:
        logging.info("Starting a new episode")
        self._environment.reset()
        self._agent.reset()
        self._subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1.0 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            self._step()
            self._episode_steps += 1

            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
        
        logging.info("Episode ended")
        self._subscriber.on_episode_end()
    
    def _step(self) -> None:
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)
        self._subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps):
            self._in_episode = False
            logging.info("Episode complete")

