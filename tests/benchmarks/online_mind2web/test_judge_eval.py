from unittest.mock import MagicMock, patch

import pytest


class TestJudgeEval:

    def test_identify_key_points(self):
        model = MagicMock()
        model.return_value = {
            "role": "assistant",
            "content": "**Key Points**:\n1. Point one.\n2. Point two.",
        }
        from agentlab.benchmarks.online_mind2web.judge_eval import identify_key_points

        task_instruction = "Test task instruction."
        key_points = identify_key_points(task_instruction, model)
        assert "Point one" in key_points
        assert "Point two" in key_points

    def test_judge_image(self):
        model = MagicMock()
        model.return_value = {
            "role": "assistant",
            "content": "**Score**: 4\n\n**Reasoning**: The screenshot shows the correct action.",
        }
        import numpy as np
        from PIL import Image

        from agentlab.benchmarks.online_mind2web.judge_eval import judge_image

        task_instruction = "Test task instruction."
        screenshot = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB")
        key_points = "1. Point one.\n2. Point two."

        response = judge_image(task_instruction, screenshot, key_points, model)
        assert "**Score**" in response
        assert "**Reasoning**" in response

    @patch(
        "agentlab.benchmarks.online_mind2web.judge_eval.encode_image",
        return_value="base64encodedstring",
    )
    @patch(
        "agentlab.benchmarks.online_mind2web.judge_eval.identify_key_points",
        return_value="1. Point one.\n2. Point two.",
    )
    @patch(
        "agentlab.benchmarks.online_mind2web.judge_eval.judge_image",
        return_value="**Score**: 4\n\n**Reasoning**: The screenshot shows the correct action.",
    )
    def test_webjudge_online_mind2web_eval_with_image(
        self, mock_encode_image, mock_identify_key_points, mock_judge_image
    ):
        import numpy as np
        from PIL import Image

        from agentlab.benchmarks.online_mind2web.judge_eval import (
            webjudge_online_mind2web_eval,
        )

        task_instruction = "Test task instruction."
        screenshots = [
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB") for _ in range(3)
        ]
        last_actions = ["Action 1", "Action 2", "Action 3"]
        model = MagicMock()
        score_threshold = 3

        prompts, user_msg, system_msg, record, key_points = webjudge_online_mind2web_eval(
            task_instruction, screenshots, last_actions, model, score_threshold
        )
        assert key_points in user_msg
        assert len(prompts) == 2  # 1 for user_msg and 1 for images
        assert mock_encode_image.call_count == 3
        assert mock_judge_image.call_count == 3
        assert mock_identify_key_points.call_count == 1

    @pytest.mark.xfail(
        reason="This test currently fails, an image is still being passed to the judge."
    )
    @patch(
        "agentlab.benchmarks.online_mind2web.judge_eval.encode_image",
        return_value="base64encodedstring",
    )
    @patch(
        "agentlab.benchmarks.online_mind2web.judge_eval.identify_key_points",
        return_value="1. Point one.\n2. Point two.",
    )
    @patch(
        "agentlab.benchmarks.online_mind2web.judge_eval.judge_image",
        return_value="Score: 4\n\nReasoning: The screenshot shows the correct action.",
    )
    def test_webjudge_online_mind2web_eval_no_image(
        self, mock_encode_image, mock_identify_key_points, mock_judge_image
    ):
        import numpy as np
        from PIL import Image

        from agentlab.benchmarks.online_mind2web.judge_eval import (
            webjudge_online_mind2web_eval,
        )

        task_instruction = "Test task instruction."
        screenshots = [
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB") for _ in range(3)
        ]
        last_actions = ["Action 1", "Action 2", "Action 3"]
        model = MagicMock()
        score_threshold = 5

        prompts, user_msg, system_msg, record, key_points = webjudge_online_mind2web_eval(
            task_instruction, screenshots, last_actions, model, score_threshold
        )
        assert key_points in user_msg
        assert len(prompts) == 1  # 1 for user_msg and 1 for images
