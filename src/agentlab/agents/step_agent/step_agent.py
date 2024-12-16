import re
from typing import Union, List

from langchain_openai import ChatOpenAI, AzureChatOpenAI

from .prompt_agent import PromptAgent
from .base_agent import BaseAgent
from .utils.stack import Stack, Element


class StepAgent(BaseAgent):
    """Adapted from https://github.com/asappresearch/webagents-step/blob/main/src/webagents_step/agents/step_agent.py"""
    
    WEBARENA_AGENTS = {
        8023: "github_agent",
        9999: "reddit_agent",
        7770: "shopping_agent",
        7780: "shopping_admin_agent",
        3000: "maps_agent",
    }

    def __init__(self, 
                 benchmark: str,
                 model: Union[ChatOpenAI, AzureChatOpenAI],
                 max_actions: int = 10, verbose: int = 0, logging: bool = False,
                 low_level_action_list: List = None,
                 prompt_mode: str = "chat",
                 previous_actions: List = None):
        super().__init__(
            max_actions=max_actions,
            previous_actions=previous_actions,
        )
        self.benchmark = benchmark
        self.root_action = None
        self.action_to_prompt_dict = {} 
        self.low_level_action_list = [] if low_level_action_list is None else low_level_action_list
        self.model = model
        self.prompt_mode = prompt_mode
        self.stack = Stack()
        self.prev_url:  str = None

    def is_done(self, action: str) -> bool:
        if "stop" in action:
            return True
        return False

    def is_low_level_action(self, action: str) -> bool:
        action_type = action.split()[0]
        return (action_type in self.low_level_action_list)

    def is_high_level_action(self, action: str) -> bool:
        action_type = action.split()[0]
        return (action_type in self.action_to_prompt_dict)

    def init_root_agent(self, objective: str) -> Element:
        root_prompt_template = self.action_to_prompt_dict[self.root_action]
        agent = PromptAgent(
            prompt_template=root_prompt_template,
            model=self.model,
            prompt_mode=self.prompt_mode,
            max_actions=self.max_actions,
            verbose=self.verbose,
            logging=self.logging,
            previous_actions=[],
            previous_reasons=[],
            previous_responses=[]
        )
        return Element(agent=agent, objective=objective)

    def init_agent(self, action: str)-> Element:
        pattern = r'(\w+)\s+\[(.*?)\]'
        matches = re.findall(pattern, action)
        action_type, _ = matches[0]
        objective = action
        prompt_template = self.action_to_prompt_dict[action_type]
        agent = PromptAgent(
            prompt_template=prompt_template,
            model=self.model,
            prompt_mode=self.prompt_mode,
            max_actions=self.max_actions,
            verbose=self.verbose,
            logging=self.logging,
            previous_actions=[],
            previous_reasons=[],
            previous_responses=[]
        )
        return Element(agent=agent, objective=objective)

    def init_actions(self, benchmark: str, url: str):
        match benchmark:
            case "miniwob":
                from .prompts.miniwob import step_fewshot_template
                self.root_action = "miniwob_agent"
            case "webarena":
                match_url = re.search(r'http:\/{2}\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}.(\d*)\/.*', url)
                port = int(match_url.group(1)) if match_url else None

                from .prompts.webarena import step_fewshot_template
                self.root_action = self.WEBARENA_AGENTS[port] if port in self.WEBARENA_AGENTS.keys() else None

                print(self.root_action)

        self.action_to_prompt_dict = {
        k: v for k, v in step_fewshot_template.__dict__.items() if isinstance(v, dict)}

    def predict_action(self, objective: str, observation: str, url: str = None) -> tuple[str, dict]: 
        if (self.root_action is None) or (url != self.prev_url):
            # Dynamically select agent policy according to the current website (useful or webarena).
            self.init_actions(self.benchmark, url)
            self.prev_url = url

        if self.stack.is_empty():
            new_element = self.init_root_agent(objective=objective)
            self.stack.push(new_element)

        action, reason = None, None
        while not self.stack.is_empty():
            element = self.stack.peek()
            action, reason = element.agent.predict_action(
                objective=element.objective, observation=observation, url=url)
            if (not self.is_done(action)) and self.is_low_level_action(action):
                element.agent.receive_response("")
                return action, reason
            if (not self.is_done(action)) and self.is_high_level_action(action):
                new_element = self.init_agent(action)
                self.stack.push(new_element)
                if self.logging:
                    self.log_step(objective=element.objective, url=url,
                                  observation=observation, action=action, reason=reason, status={})
                continue
            if self.is_done(action):
                self.stack.pop()
                if not self.stack.is_empty():
                    self.stack.peek().agent.receive_response(
                        re.search(r"\[(.*?)\]", action).group(1))
                if self.logging:
                    self.log_step(objective=element.objective, url=url,
                                  observation=observation, action=action, reason=reason, status={})
                continue
        return action, reason
