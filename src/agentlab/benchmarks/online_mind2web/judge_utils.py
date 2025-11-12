"""
Copied from https://github.com/OSU-NLP-Group/Online-Mind2Web/blob/main/src/utils.py
"""

import base64
import io


def encode_image(image):
    """Convert a PIL image to base64 string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_prediction(response, mode):
    """Extract the prediction from the response."""
    if mode == "Autonomous_eval":
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "AgentTrek_eval":
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "WebVoyager_eval":
        if "FAILURE" in response:
            return 0
        else:
            return 1
    elif mode == "WebJudge_Online_Mind2Web_eval":
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "WebJudge_general_eval":
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            else:
                return 0
        except:
            return 0
    else:
        raise ValueError(f"Unknown mode: {mode}")
