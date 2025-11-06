
from dataclasses import dataclass
from typing import Any

import playwright.sync_api
from browsergym.core.env import AbstractBrowserTask

from agentlab.llm.chat_api import BaseModelArgs

from .judge_eval import WebJudge_Online_Mind2Web_eval
from .judge_utils import extract_prediction


@dataclass
class JudgeInfo:
    chat_messages: list[str]
    record: str
    key_points: str
    reward: int


@dataclass
class OnlineMind2WebTaskConfig:
    task_id: str
    confirmed_task: str
    website: str
    reference_length: int
    level: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OnlineMind2WebTaskConfig":
        return cls(
            task_id=data["task_id"],
            confirmed_task=data["confirmed_task"],
            website=data["website"],
            reference_length=data["reference_length"],
            level=data["level"],
        )
    

class OnlineMind2WebTask(AbstractBrowserTask):
    def __init__(
            self, 
            seed: int,
            task_config: OnlineMind2WebTaskConfig,
            judge_model_args: BaseModelArgs = None,
            judge_score_threshold: int = 3
        ) -> None:
        """
        Args:
            task: str, the task instruction.

        """
        super().__init__(seed=0)
        self.task_config = task_config
        self.judge_score_threshold = judge_score_threshold  # Threshold for considering a screenshot relevant to judge the task completion. Scores are between 1 and 5.

        self.judge_model = judge_model_args.make_model()
        self.action_history = []
        self.screenshots = []

        # Task timeout
        self.timeout = 15 * 60  # 15 minutes

    def setup(self, page):
        # Navigate to the starting URL
        if self.task_config.website:
            start_url = self.task_config.website.strip()
            page.goto(start_url, timeout=60000)  # 60 second timeout
        
        return self.task_config.confirmed_task, {}

    def validate(self, page: playwright.sync_api.Page, chat_messages: list[str]):
        judge_prompt_messages, user_msg, system_msg, record, key_points = WebJudge_Online_Mind2Web_eval(
            task_instruction=self.task_config.confirmed_task,
            last_actions=self.action_history,
            screenshots=self.screenshots,
            model=self.judge_model,
            score_threshold=self.judge_score_threshold,
        )

        response = self.judge_model(judge_prompt_messages)["content"]

        reward = extract_prediction(response, mode="WebJudge_Online_Mind2Web_eval")

        # Stop task when the agent sends a message to the user
        last_action_is_stop = (
            chat_messages and chat_messages[-1]["role"] == "assistant" \
            and "send_msg_to_user" in chat_messages[-1]["message"].strip().lower() if "message" in chat_messages[-1] else False
        )

        if reward > 0 or last_action_is_stop:
            done, user_message, info = True, None, {}
        else:
            done, user_message, info = False, "", {}

        return reward, done, user_message, info
    
    def cheat(self, page, chat_messages):
        pass

    def teardown(self):
        # Nothing to clean up
        pass
