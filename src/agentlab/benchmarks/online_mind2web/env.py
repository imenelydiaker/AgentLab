from dataclasses import dataclass, field

from browsergym.core.env import BrowserEnv
from PIL import Image

from agentlab.benchmarks.abstract_env import (
    AbstractEnv,
    AbstractEnvArgs,
)
from agentlab.llm.chat_api import BaseModelArgs, OpenAIModelArgs

from .task import OnlineMind2WebTask, OnlineMind2WebTaskConfig


# Default judge model arguments factory
# Authors recommend to use "o4-mini" for judging OnlineMind2Web tasks
def _default_judge_model_args() -> OpenAIModelArgs:
    return OpenAIModelArgs(
        model_name="o4-mini",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=10_000,
        vision_support=True,
    )

@dataclass
class OnlineMind2WebEnvArgs(AbstractEnvArgs):
    task_config: OnlineMind2WebTaskConfig
    task_name: str
    action_space: str = "bid"
    max_steps: int = 30
    task_seed: int = 0
    judge_model_args: BaseModelArgs = None # field(default_factory=_default_judge_model_args)  # Default judge model is set to o4-mini
    judge_score_threshold: int = 3

    def make_env(self, action_mapping, exp_dir, **exp_task_kwargs) -> "OnlineMind2WebEnv":
        """Create an OnlineMind2Web environment instance."""
        env = OnlineMind2WebEnv(
            task_config=self.task_config,
            task_name=self.task_name,
            action_mapping=action_mapping,
            exp_dir=exp_dir,
            max_steps=self.max_steps,
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
        judge_model_args: BaseModelArgs = None,
        judge_score_threshold: int = 3,
        **kwargs
    ):
        self.task_config = task_config
        self.task_name = task_name
        self.exp_dir = exp_dir
        self.max_steps = max_steps

        # Store reference to episode_info list from the experiment loop
        # This will be updated by the loop as the episode progresses
        self.action_history = []
        self.screenshots = []
        
        # Prepare task_kwargs for BrowserEnv
        # BrowserEnv will call OnlineMind2WebTask(seed=seed, **task_kwargs)
        task_kwargs = {
            "task_config": task_config,
            "judge_model_args": judge_model_args,
            "judge_score_threshold": judge_score_threshold,
        }
        
        # Initialize BrowserEnv with the task class (not an instance)
        BrowserEnv.__init__(
            self,
            task_entrypoint=OnlineMind2WebTask,
            task_kwargs=task_kwargs,
            **kwargs,
        )

    def set_episode_info(self, episode_info: list):
        """Set the episode_info reference from the experiment loop."""
        self.action_history = [info.action for info in episode_info]
        self.screenshots = [Image.fromarray(info.obs["screenshot"], "RGB") for info in episode_info]
    
    def reset(self, seed: int = None):
        return BrowserEnv.reset(self, seed=seed)
    
    def step(self, action: str):
        # Update the task's action history and screenshots before stepping
        self.task.action_history = self.action_history
        self.task.screenshots = self.screenshots
        return BrowserEnv.step(self, action)
    
    def close(self):
        return BrowserEnv.close(self)

