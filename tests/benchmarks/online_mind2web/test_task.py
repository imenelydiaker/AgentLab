import pytest


class TestOnlineMind2WebTask:
    def test_task_initialization(self):
        from unittest.mock import MagicMock

        from agentlab.benchmarks.online_mind2web.task import (
            OnlineMind2WebTask,
            OnlineMind2WebTaskConfig,
        )

        task = OnlineMind2WebTask(
            seed=0,
            task_config=OnlineMind2WebTaskConfig(
                task_id="test_task_001",
                confirmed_task="Find the official website of OpenAI.",
                website="https://www.openai.com",
                reference_length=5,
                level="easy"
            ),
            judge_model_args=MagicMock()
        )

        assert task.task_config.task_id == "test_task_001"
        assert task.task_config.confirmed_task == "Find the official website of OpenAI."
        assert task.task_config.website == "https://www.openai.com"
        assert task.task_config.reference_length == 5
        assert task.task_config.level == "easy"

    def test_validate(self):
        from unittest.mock import MagicMock, patch

        import numpy as np
        from PIL import Image

        from agentlab.benchmarks.online_mind2web.task import (
            OnlineMind2WebTask,
            OnlineMind2WebTaskConfig,
        )
        
        # Create a task config
        task_config = OnlineMind2WebTaskConfig(
            task_id="test_task_001",
            confirmed_task="Find the official website of OpenAI.",
            website="https://www.openai.com",
            reference_length=5,
            level="easy"
        )
        
        # Create mock judge model
        mock_judge_model = MagicMock()
        mock_judge_model.return_value = {"content": "The task is completed successfully. Status: SUCCESS"}
        
        # Create mock judge model args
        mock_judge_model_args = MagicMock()
        mock_judge_model_args.make_model.return_value = mock_judge_model
        
        # Create task instance
        task = OnlineMind2WebTask(
            seed=42,
            task_config=task_config,
            judge_model_args=mock_judge_model_args,
            judge_score_threshold=3
        )
        
        # Set up action history and screenshots
        task.action_history = ["click", "type", "scroll"]
        task.screenshots = [
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB"),
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB"),
        ]
        
        # Mock the webjudge_online_mind2web_eval function
        with patch('agentlab.benchmarks.online_mind2web.task.webjudge_online_mind2web_eval') as mock_eval:
            mock_eval.return_value = (
                [{"role": "user", "content": "test"}],  # judge_prompt_messages
                "user_msg",  # user_msg
                "system_msg",  # system_msg
                "record",  # record
                "key_points"  # key_points
            )
            
            # Mock page and chat_messages
            mock_page = MagicMock()
            chat_messages = [
                {"role": "user", "content": "Do the task"},
                {"role": "assistant", "content": "Working on it"}
            ]
            
            # Call validate
            reward, done, user_message, info = task.validate(mock_page, chat_messages)
            
            # Verify webjudge_online_mind2web_eval was called with correct arguments
            mock_eval.assert_called_once_with(
                task_instruction="Find the official website of OpenAI.",
                last_actions=["click", "type", "scroll"],
                screenshots=task.screenshots,
                model=mock_judge_model,
                score_threshold=3,
            )
            
            # Verify judge model was called
            mock_judge_model.assert_called_once()
            
            # Verify reward is 1 (success)
            assert reward == 1
            
            # Verify task is done
            assert done is True
            
            # Verify user_message is None (task completed)
            assert user_message is None
            
            # Verify info is empty dict
            assert info == {}
    
    def test_validate_failure(self):
        from unittest.mock import MagicMock, patch

        import numpy as np
        from PIL import Image

        from agentlab.benchmarks.online_mind2web.task import (
            OnlineMind2WebTask,
            OnlineMind2WebTaskConfig,
        )
        
        # Create a task config
        task_config = OnlineMind2WebTaskConfig(
            task_id="test_task_002",
            confirmed_task="Complete a complex task.",
            website="https://example.com",
            reference_length=20,
            level="hard"
        )
        
        # Create mock judge model that returns failure
        mock_judge_model = MagicMock()
        mock_judge_model.return_value = {"content": "The task failed. Status: FAILURE"}
        
        # Create mock judge model args
        mock_judge_model_args = MagicMock()
        mock_judge_model_args.make_model.return_value = mock_judge_model
        
        # Create task instance
        task = OnlineMind2WebTask(
            seed=42,
            task_config=task_config,
            judge_model_args=mock_judge_model_args,
            judge_score_threshold=3
        )
        
        # Set up action history and screenshots
        task.action_history = ["click"]
        task.screenshots = [
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB"),
        ]
        
        # Mock the webjudge_online_mind2web_eval function
        with patch('agentlab.benchmarks.online_mind2web.task.webjudge_online_mind2web_eval') as mock_eval:
            mock_eval.return_value = (
                [{"role": "user", "content": "test"}],
                "user_msg",
                "system_msg",
                "record",
                "key_points"
            )
            
            # Mock page and chat_messages
            mock_page = MagicMock()
            chat_messages = [
                {"role": "user", "content": "Do the task"},
                {"role": "assistant", "content": "Working on it"}
            ]
            
            # Call validate
            reward, done, user_message, info = task.validate(mock_page, chat_messages)
            
            # Verify reward is 0 (failure)
            assert reward == 0
            
            # Verify task is not done
            assert done is False
            
            # Verify user_message is empty string
            assert user_message == ""
            
            # Verify info is empty dict
            assert info == {}
    
    def test_validate_with_stop_action(self):
        from unittest.mock import MagicMock, patch

        import numpy as np
        from PIL import Image

        from agentlab.benchmarks.online_mind2web.task import (
            OnlineMind2WebTask,
            OnlineMind2WebTaskConfig,
        )
        
        # Create a task config
        task_config = OnlineMind2WebTaskConfig(
            task_id="test_task_003",
            confirmed_task="Send a message to user.",
            website="https://example.com",
            reference_length=5,
            level="easy"
        )
        
        # Create mock judge model that returns failure
        mock_judge_model = MagicMock()
        mock_judge_model.return_value = {"content": "Task not completed. Status: FAILURE"}
        
        # Create mock judge model args
        mock_judge_model_args = MagicMock()
        mock_judge_model_args.make_model.return_value = mock_judge_model
        
        # Create task instance
        task = OnlineMind2WebTask(
            seed=42,
            task_config=task_config,
            judge_model_args=mock_judge_model_args,
            judge_score_threshold=3
        )
        
        # Set up action history and screenshots
        task.action_history = ["send_msg_to_user"]
        task.screenshots = [
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB"),
        ]
        
        # Mock the webjudge_online_mind2web_eval function
        with patch('agentlab.benchmarks.online_mind2web.task.webjudge_online_mind2web_eval') as mock_eval:
            mock_eval.return_value = (
                [{"role": "user", "content": "test"}],
                "user_msg",
                "system_msg",
                "record",
                "key_points"
            )
            
            # Mock page and chat_messages with stop action
            mock_page = MagicMock()
            chat_messages = [
                {"role": "user", "content": "Do the task"},
                {"role": "assistant", "message": "send_msg_to_user: Done!"}
            ]
            
            # Call validate
            reward, done, user_message, info = task.validate(mock_page, chat_messages)
            
            # Verify reward is 0 (failure from judge)
            assert reward == 0
            
            # Verify task is done due to stop action
            assert done is True
            
            # Verify user_message is None
            assert user_message is None
            
            # Verify info is empty dict
            assert info == {}

    def test_setup(self):
        from unittest.mock import MagicMock

        from agentlab.benchmarks.online_mind2web.task import (
            OnlineMind2WebTask,
            OnlineMind2WebTaskConfig,
        )
        
        # Create a task config
        task_config = OnlineMind2WebTaskConfig(
            task_id="test_task_001",
            confirmed_task="Find the official website of OpenAI.",
            website="https://www.openai.com",
            reference_length=5,
            level="easy"
        )
        
        # Create task instance
        task = OnlineMind2WebTask(
            seed=42,
            task_config=task_config,
            judge_model_args=MagicMock(),
            judge_score_threshold=3
        )
        
        # Mock the page object
        mock_page = MagicMock()
        
        # Call setup
        goal, info = task.setup(mock_page)
        
        # Verify that set_default_timeout was called with 30000ms
        mock_page.set_default_timeout.assert_called_once_with(30000)
        
        # Verify that goto was called with the website URL
        mock_page.goto.assert_called_once_with("https://www.openai.com", wait_until="load")
        
        # Verify that the returned goal is the confirmed_task
        assert goal == "Find the official website of OpenAI."
        
        # Verify that info is an empty dict
        assert info == {}
    
    def test_setup_without_website(self):
        from unittest.mock import MagicMock

        from agentlab.benchmarks.online_mind2web.task import (
            OnlineMind2WebTask,
            OnlineMind2WebTaskConfig,
        )
        
        # Create a task config without a website
        task_config = OnlineMind2WebTaskConfig(
            task_id="test_task_002",
            confirmed_task="Complete a task without initial URL.",
            website="",
            reference_length=10,
            level="medium"
        )
        
        # Create task instance
        task = OnlineMind2WebTask(
            seed=42,
            task_config=task_config,
            judge_model_args=MagicMock(),
            judge_score_threshold=3
        )
        
        # Mock the page object
        mock_page = MagicMock()
        
        # Call setup
        goal, info = task.setup(mock_page)
        
        # Verify that set_default_timeout was called
        mock_page.set_default_timeout.assert_called_once_with(30000)
        
        # Verify that goto was NOT called since website is empty
        mock_page.goto.assert_not_called()
        
        # Verify that the returned goal is still the confirmed_task
        assert goal == "Complete a task without initial URL."
        
        # Verify that info is an empty dict
        assert info == {}
        