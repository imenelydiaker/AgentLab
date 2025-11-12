import pytest


class TestOnlineMind2WebEnv:
    def test_extract_episode_info(self):
        import numpy as np
        from PIL import Image

        from agentlab.benchmarks.online_mind2web.env import OnlineMind2WebEnv
        from agentlab.experiments.loop import StepInfo

        env = OnlineMind2WebEnv(
            task_config=None,
            task_name="test_task_name",
            action_mapping={},
            exp_dir="/tmp/test_exp_dir",
        )

        # Mock a response with episode info
        mock_response = [
            StepInfo(action="click", obs={"screenshot": np.zeros((100, 100, 3), dtype=np.uint8)}),
            StepInfo(action="type", obs={"screenshot": np.zeros((100, 100, 3), dtype=np.uint8)}),
        ]

        env.extract_episode_info(mock_response)
        assert env.action_history == ["click", "type"]
        assert all(isinstance(img, Image.Image) for img in env.screenshots)
        assert len(env.screenshots) == 2
