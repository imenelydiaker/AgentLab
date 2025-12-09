from dataclasses import dataclass

from browsergym.core.action.base import execute_python_code
from browsergym.core.env import BrowserEnv
from PIL import Image

from agentlab.benchmarks.abstract_env import (
    AbstractEnv,
    AbstractEnvArgs,
)
from agentlab.llm.chat_api import BaseModelArgs

from .task import OnlineMind2WebTask, OnlineMind2WebTaskConfig


@dataclass
class OnlineMind2WebEnvArgs(AbstractEnvArgs):
    task_config: OnlineMind2WebTaskConfig
    task_name: str
    action_space: str = "bid"
    max_steps: int = 30
    validate_at_each_step: bool = False
    task_seed: int = 0
    judge_model_args: BaseModelArgs = None
    judge_score_threshold: int = 3

    def make_env(self, action_mapping, exp_dir, **exp_task_kwargs) -> "OnlineMind2WebEnv":
        """Create an OnlineMind2Web environment instance."""
        env = OnlineMind2WebEnv(
            task_config=self.task_config,
            task_name=self.task_name,
            action_mapping=action_mapping,
            exp_dir=exp_dir,
            max_steps=self.max_steps,
            validate_at_each_step=self.validate_at_each_step,
            judge_model_args=self.judge_model_args,
            judge_score_threshold=self.judge_score_threshold,
            **exp_task_kwargs,
        )
        return env


class OnlineMind2WebEnv(AbstractEnv, BrowserEnv):
    """
    Environment for OnlineMind2Web tasks.
    This environment inherits from browsergym.core.env.BrowserEnv
    and integrates a method to get episode information from the
    interaction loop. The episode information is used to share
    the action history and screenshots with the task instance.
    """

    def __init__(
        self,
        task_config: OnlineMind2WebTaskConfig,
        task_name: str,
        action_mapping,
        exp_dir,
        max_steps: int = 30,
        validate_at_each_step: bool = False,
        judge_model_args: BaseModelArgs = None,
        judge_score_threshold: int = 3,
        **kwargs,
    ):
        self.task_config = task_config
        self.task_name = task_name
        self.exp_dir = exp_dir
        self.max_steps = max_steps
        self.validate_at_each_step = validate_at_each_step

        # Prepare task_kwargs for BrowserEnv
        # BrowserEnv will call OnlineMind2WebTask(seed=seed, **task_kwargs)
        task_kwargs = {
            "task_config": task_config,
            "judge_model_args": judge_model_args,
            "judge_score_threshold": judge_score_threshold,
            "validate_at_each_step": validate_at_each_step,
        }

        # Initialize BrowserEnv with the task class (not an instance)
        BrowserEnv.__init__(
            self,
            task_entrypoint=OnlineMind2WebTask,
            task_kwargs=task_kwargs,
            **kwargs,
        )

    def collect_episode_info(self, episode_info: list):
        """This method is called by the experiment loop to provide
        the environment with the current episode information (episode_info).
        """
        self.task.action_history = [info.action for info in episode_info]
        self.task.screenshots = [Image.fromarray(info.obs["screenshot"], "RGB") for info in episode_info]

    def reset(self, seed: int = None):
        self._step_count = 0
        return BrowserEnv.reset(self, seed=seed)

    def step(self, action):
        self._step_count += 1
        obs, reward, terminated, truncated, info = BrowserEnv.step(self, action)
        if self._step_count >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

    def close(self):
        return BrowserEnv.close(self)
