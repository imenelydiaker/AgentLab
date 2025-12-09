import logging
from dataclasses import dataclass
from typing import Any

import playwright.sync_api
from browsergym.core.env import AbstractBrowserTask

from agentlab.llm.chat_api import BaseModelArgs

from .judge_eval import webjudge_online_mind2web_eval
from .judge_utils import extract_prediction

logger = logging.getLogger(__name__)


@dataclass
class OnlineMind2WebTaskConfig:
    task_id: str
    confirmed_task: str
    website: str
    reference_length: int
    level: str


class OnlineMind2WebTask(AbstractBrowserTask):
    def __init__(
        self,
        seed: int,
        task_config: OnlineMind2WebTaskConfig,
        judge_model_args: BaseModelArgs = None,
        judge_score_threshold: int = 3,
        validate_at_each_step: bool = False,
    ) -> None:
        """
        Args:
            task: str, the task instruction.

        """
        super().__init__(seed)
        self.task_config = task_config
        self.judge_score_threshold = judge_score_threshold  # Threshold for considering a screenshot relevant to judge the task completion. Scores are between 1 and 5.
        self.validate_at_each_step = validate_at_each_step

        self.judge_model = judge_model_args.make_model()
        self.action_history = []
        self.screenshots = []

        # Task timeout
        self.timeout = 15 * 60  # 15 minutes

    def setup(self, page):
        page.set_default_timeout(60000)  # applies to all waits and navigations

        # Navigate to the starting URL
        if self.task_config.website:
            start_url = self.task_config.website.strip()
            page.goto(start_url, timeout=60000)  # Wait up to 60 seconds for the page to load

        return self.task_config.confirmed_task, {}

    def agent_is_done(self):
        # Check if the last action is sending a message to the user
        return "send_msg_to_user" in self.action_history[-1]
        
    def validate(self, page: playwright.sync_api.Page, chat_messages: list[str]):
        # Stop task when the agent sends a message to the user
        done = self.agent_is_done()
        reward = 0

        if self.validate_at_each_step or done:
            logger.info(
                "Validating task completion with judge model. Task ID: %s", self.task_config.task_id
            )
            judge_prompt_messages, user_msg, system_msg, record, key_points = (
                webjudge_online_mind2web_eval(
                    task_instruction=self.task_config.confirmed_task,
                    last_actions=self.action_history,
                    screenshots=self.screenshots,
                    model=self.judge_model,
                    score_threshold=self.judge_score_threshold,
                )
            )

            response = self.judge_model(judge_prompt_messages)["content"]
            logger.info(f"Judge response: {response}")

            reward = extract_prediction(response, mode="WebJudge_Online_Mind2Web_eval")

        logger.info(f"Reward: {reward}. Agent is done: {done}")
        user_message, info = "", {}
        return reward, done, user_message, info

    def cheat(self, page, chat_messages):
        pass

    def teardown(self):
        # Nothing to clean up
        pass
