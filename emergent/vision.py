import torch, os
from config import load_environ

from image_analysis import AnalyzeImage
from util import extract_frame, resize_image, cleanup

class VisibilityChecker:
    def __init__(self, visual_id, width, height):
        self.visual_id = visual_id
        self.width = width
        self.height = height

    def check(self, media_path, output_dir):
        """Returns: (is_visible: bool, reason_code: str, analysis: str)"""

        frame, check_path = extract_frame(media_path, self.width, self.height, 
                                f"{output_dir}/check_vision.png")
        
        # Tier 2: VLM Orientation Check
        prompt = f"""You are an expert visual evaluator. Analyze the image to check for a specific character and determine their orientation.

        Target Character Description: {self.visual_id}

        Step 1: Identity Check
        Is there a character visible that generally matches the target description?

        Step 2: Visual Analysis & Orientation
        Determine the character's orientation based on these definitions:
        - "facing_camera": Front or 3/4 view, face clearly visible with eyes discernible.
        - "profile_engaged": 3/4 or side profile, face partially visible (at least one eye visible), clearly engaged in a task.
        - "looking_down_task": Head pitched downward, but face still partially visible (profile angle), actively working on object directly in front.
        - "walking_away": Showing back, actively moving away from camera.
        - "turned_away": Side or back view, face NOT visible, eyes NOT visible.
        - "partially_visible": Heavily obscured, extreme angle, or face completely turned away.

        CRITICAL RULE: If the face is not visible or eyes cannot be discerned (even partially), classify as "turned_away" or "partially_visible". Do NOT accept "looking down" or "engaged in task" as valid if the face is completely obscured.

        Output format (STRICTLY follow this exact order and format):
        MATCH: [YES/NO]
        ANALYSIS: [1-2 sentences describing their physical action, head pose, and whether face/eyes are visible.]
        ORIENTATION: [facing_camera / profile_engaged / looking_down_task / walking_away / turned_away / partially_visible]
        """
        
        result = AnalyzeImage(check_path, prompt)
        response = result['analysis'].strip()
        
        return self._parse_vision_response(response)

    def _parse_vision_response(self, response):
        match, orientation, analysis = "NO", "unknown", "Unknown"
        for line in response.split('\n'):
            line = line.strip()
            if line.upper().startswith("MATCH:"):
                match = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("ORIENTATION:"):
                orientation = line.split(":", 1)[1].strip().lower().strip('."\'')
            elif line.upper().startswith("ANALYSIS:"):
                analysis = line.split(":", 1)[1].strip()

        if match == "YES" and orientation in ["facing_camera", "profile_engaged", "looking_down_task"]:
            return True, "visible", analysis
        elif orientation in ["walking_away", "turned_away", "partially_visible"]:
            return False, orientation, analysis
        elif match == "NO":
            return False, "wrong_character", analysis
        else:
            return False, "unknown", analysis