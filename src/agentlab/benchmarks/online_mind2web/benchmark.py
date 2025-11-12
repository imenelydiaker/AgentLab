import json
import os
import logging
from pathlib import Path
from typing import Any, Literal

from browsergym.experiments.benchmark.base import HighLevelActionSetArgs
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS

from agentlab.benchmarks.abstract_env import AbstractBenchmark
from agentlab.llm.chat_api import BaseModelArgs

from .env import OnlineMind2WebEnvArgs
from .task import OnlineMind2WebTaskConfig

logger = logging.getLogger(__name__)


class OnlineMind2WebBenchmark(AbstractBenchmark):
    name: str = "online_mind2web"
    is_multi_tab: bool = True
    high_level_action_set_args: HighLevelActionSetArgs = DEFAULT_HIGHLEVEL_ACTION_SET_ARGS[
        "webarena"
    ]  # similar to webarena
    judge_model_args: BaseModelArgs = None
    judge_score_threshold: int = 3
    validate_at_each_step: bool = False
    level: Literal["easy", "medium", "hard", "all"] = "all"
    task_file: str = None  # type: ignore
    env_args_list: list["OnlineMind2WebEnvArgs"] = None  # type: ignore

    def __post_init__(self):
        if self.level != "all":
            self.name += f"_{self.level}"

    def model_post_init(self, __context: Any) -> None:
        """Load tasks from the JSON file and create env_args_list."""
        self.env_args_list = []

        # Determine the path to the JSON file
        if self.task_file is None:
            # Default to the JSON file in the online_mind2web directory
            current_dir = Path(__file__).parent
            self.task_file = str(current_dir / "config.json")

        # Load tasks from JSON
        if not os.path.exists(self.task_file):
            logger.warning(f"Task file not found: {self.task_file}")
            return

        with open(self.task_file, "r") as f:
            tasks_dict = json.load(f)

        tasks = [OnlineMind2WebTaskConfig(**item) for item in tasks_dict]

        # Filter by level if specified
        for task_config in tasks:
            if self.level != "all" and task_config.level != self.level:
                continue

            task_name = f"online_mind2web.{task_config.task_id}"

            env_args = OnlineMind2WebEnvArgs(
                task_config=task_config,
                task_name=task_name,
                judge_model_args=self.judge_model_args,
                judge_score_threshold=self.judge_score_threshold,
                validate_at_each_step=self.validate_at_each_step,
            )
            self.env_args_list.append(env_args)

        logger.info(f"Loaded {len(self.env_args_list)} OnlineMind2Web tasks (level: {self.level})")
