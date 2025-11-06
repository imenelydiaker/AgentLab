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
        assert env.action_history ==  ["click", "type"]
        assert all(isinstance(img,  Image.Image) for img in env.screenshots)
        assert len(env.screenshots) == 2 
    
    def test_step(self):
        from unittest.mock import MagicMock, Mock, patch

        import numpy as np
        from PIL import Image

        from agentlab.benchmarks.online_mind2web.env import OnlineMind2WebEnv
        from agentlab.benchmarks.online_mind2web.task import OnlineMind2WebTaskConfig
        
        # Create a mock task config
        task_config = OnlineMind2WebTaskConfig(
            task_id="test_id",
            confirmed_task="Test task",
            website="https://example.com",
            reference_length=5,
            level="easy"
        )
        
        # Create environment
        env = OnlineMind2WebEnv(
            task_config=task_config,
            task_name="test_task",
            action_mapping={},
            exp_dir="/tmp/test_exp_dir",
            max_steps=10,
            judge_model_args=None,
            judge_score_threshold=3,
        )
        
        # Mock the task object that would be created by BrowserEnv
        mock_task = MagicMock()
        mock_task.action_history = []
        mock_task.screenshots = []
        env.task = mock_task
        
        # Set up action history and screenshots in env
        env.action_history = ["click", "type", "scroll"]
        test_screenshots = [
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB"),
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB"),
        ]
        env.screenshots = test_screenshots
        
        # Mock the parent class step method
        with patch.object(type(env).__bases__[1], 'step', return_value=(Mock(), 0, False, False, {})) as mock_step:
            # Call step
            env.step("test_action")
            
            # Verify that task's action_history and screenshots were updated
            assert env.task.action_history == ["click", "type", "scroll"]
            assert env.task.screenshots == test_screenshots
            
            # Verify that parent's step was called with the action
            mock_step.assert_called_once_with(env, "test_action")
    