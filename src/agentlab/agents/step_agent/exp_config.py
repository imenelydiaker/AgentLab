import logging

from browsergym.experiments.loop import EnvArgs, ExpArgs

from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.experiments import task_collections as tasks
from agentlab.experiments import args
from agentlab.agents.step_agent.browsergym_step_agent import BrowserGymStepAgentArgs


STEP_AGENT_MINIWOB_ARGS = BrowserGymStepAgentArgs(
    agent_name="StepAgentMiniWoB",
    model=CHAT_MODEL_ARGS_DICT["azure/gpt-35-turbo-1106/gpt-35-turbo-1106"],
    low_level_action_list=["click", "type", "stop"],
    use_dom=True,
    benchmark="miniwob",
    logging=True,
)

STEP_AGENT_WEBARENA_ARGS = BrowserGymStepAgentArgs(
    agent_name="StepAgentWebArena",
    model=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    low_level_action_list=['click', 'type', 'scroll', 'stop', 'goto', 'hover', 'note', 'go_back'],
    use_dom=False,
    benchmark="webarena",
    logging=True,
)

def step_agent_webarena(agent: BrowserGymStepAgentArgs = STEP_AGENT_MINIWOB_ARGS, benchmark="webarena"):
    """Run SteP on all WebArena tasks."""
    return args.expand_cross_product(
        ExpArgs(
            agent_args=agent,
            env_args=EnvArgs(
                max_steps=5,
                task_seed=args.CrossProd([None]),
                task_name=args.CrossProd(tasks.ALL_WEBARENA_TASK_IDS),
            ),
            enable_debug=True,
        )
    )


def step_agent_test(agent: BrowserGymStepAgentArgs = STEP_AGENT_MINIWOB_ARGS, benchmark="miniwob"):
    """Minimalistic experiment to test the system."""
    return args.expand_cross_product(
        ExpArgs(
            agent_args=agent,
            env_args=EnvArgs(
                max_steps=5,
                task_seed=args.CrossProd([None] * 2),
                task_name=args.CrossProd(tasks.step_tasks),
            ),
            enable_debug=True,
        )
    )
