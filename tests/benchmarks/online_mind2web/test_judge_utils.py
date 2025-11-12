import numpy as np
import pytest
from PIL import Image


@pytest.fixture(params=["RGB", "RGBA"])
def test_image(request):
    """Fixture that provides both RGB and RGBA images."""
    if request.param == "RGB":
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        return Image.fromarray(img_array, "RGB")
    else:  # RGBA
        img_array = np.zeros((100, 100, 4), dtype=np.uint8)
        return Image.fromarray(img_array, "RGBA")


class TestJudgeUtils:
    def test_encode_image(self, test_image):
        from agentlab.benchmarks.online_mind2web.judge_utils import encode_image

        encoded_str = encode_image(test_image)
        assert isinstance(encoded_str, str)
        assert len(encoded_str) > 0

    @pytest.mark.parametrize(
        "mode,response,expected",
        [
            ("Autonomous_eval", "The task is completed successfully. Status: SUCCESS", 1),
            ("Autonomous_eval", "The task failed to complete. Status: FAILURE", 0),
            ("Autonomous_eval", "The task didn't complete", 0),
            ("AgentTrek_eval", "The task is completed successfully. Status: SUCCESS", 1),
            ("AgentTrek_eval", "The task failed to complete. Status: FAILURE", 0),
            ("AgentTrek_eval", "The task didn't complete", 0),
            ("WebVoyager_eval", "The task is completed successfully.", 1),
            ("WebVoyager_eval", "FAILURE: The task failed to complete.", 0),
            (
                "WebJudge_Online_Mind2Web_eval",
                "The task is completed successfully. Status: SUCCESS",
                1,
            ),
            ("WebJudge_Online_Mind2Web_eval", "The task failed to complete. Status: FAILURE", 0),
            ("WebJudge_Online_Mind2Web_eval", "The task didn't complete", 0),
            ("WebJudge_general_eval", "The task is completed successfully. Status: SUCCESS", 1),
            ("WebJudge_general_eval", "The task failed to complete. Status: FAILURE", 0),
            ("WebJudge_general_eval", "The task didn't complete", 0),
        ],
    )
    def test_extract_prediction(self, mode, response, expected):
        from agentlab.benchmarks.online_mind2web.judge_utils import extract_prediction

        assert extract_prediction(response, mode) == expected

    def test_extract_prediction_unknown_mode(self):
        from agentlab.benchmarks.online_mind2web.judge_utils import extract_prediction

        with pytest.raises(ValueError):
            extract_prediction(response="some response", mode="unknown")
