import torch, os
from config import load_environ

load_environ()

from uniface.detection import RetinaFace
from image_analysis import AnalyzeImage
from util import extract_frame, resize_image, cleanup

ANIME = os.environ.get("ANIME", "False") != "False"

class VisibilityChecker:
    def __init__(self, visual_id, width, height):
        self.visual_id = visual_id
        self.width = width
        self.height = height

    def check(self, media_path, output_dir):
        """Returns: (is_visible: bool, reason_code: str, analysis: str)"""

        if not ANIME:
            frame, check_path = extract_frame(media_path, self.width, self.height, 
                                            f"{output_dir}/check_vision.png")
            
            # Tier 1: Face Detection
            resized_img, _ = resize_image(frame, max_dim=640)
            detector = RetinaFace()
            with torch.no_grad():
                faces = detector.detect(resized_img) 
            del detector
            cleanup()
            
            if not faces:
                return False, "no_face", "No faces detected in frame"
        
        # Tier 2: VLM Orientation Check
        prompt = f"""You are an expert visual evaluator. Analyze the image to check for a specific character and determine their orientation.

Target Character Description: {self.visual_id}

Step 1: Identity Check
Is there a character visible that generally matches the target description?

Step 2: Visual Analysis & Orientation
Determine the character's orientation based on these definitions:
- "facing_camera": Front or 3/4 view, face clearly visible.
- "profile_engaged": 3/4 or side profile, clearly engaged in a task (e.g., cooking, looking down) OR interacting with someone. (VALID STATE)
- "looking_down_task": Head pitched downward, focused on an object/task. Eyes may be obscured by angle or natural anatomy (e.g., epicanthic folds), but posture shows active attention. (VALID STATE)
- "walking_away": Showing back, actively moving away.
- "turned_away": Side/back view, facing away with NO clear task or focus (disengaged).
- "partially_visible": Heavily obscured or extreme angle where identity is lost.

CRITICAL RULE: If the character is looking down at a task or in profile but engaged, DO NOT classify them as "turned_away", even if their eyes are hidden by angle or anatomy.

Output format (STRICTLY follow this exact order and format):
MATCH: [YES/NO]
ANALYSIS: [1-2 sentences describing their physical action, head pose, and what they are looking at. Explicitly state if eyes are obscured by downward angle/anatomy.]
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