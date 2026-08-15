from plan10.lib.image_analysis import AnalyzeMedia
from plan10.lib.qwen_llm import llm_analyze_media
from pathlib import Path
from plan10.lib.config import load_config
load_config()

import os

WGP = os.environ.get("WGP","False") != "False"
LTX = os.environ.get("LTX", "False") != "False"
MMH3 = os.environ.get("MMH3","False") != "False"
DIALOG_ALLOWED = WGP or LTX or MMH3
class Director:
    def analyze_reality(self, media_path, intended_action, width, height, output_dir):
        """Analyze what actually happened in the video/image."""
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        
        # SmolVLM2 can analyze videos directly
        prompt = f"""We intended to create this: "{intended_action}"

Analyze what ACTUALLY happened in this {"video" if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm'] else "image"}:
1. What is the character doing? (actions, expressions, movement)
2. What props/objects are visible?
3. Any issues or unexpected elements?

Be factual about what you see, not what was intended."""
        
        # Use AnalyzeMedia which will use SmolVLM2 for video, Qwen for images
        result = AnalyzeMedia(str(media_path), prompt)
        return self._clean_analysis(result)

    def compare_and_decide(self, intended_action, actual_reality, story_context, history, pending_setup, goal=None, force_transition=False, location_constraint=None):
        if DIALOG_ALLOWED:
            return self.compare_and_decide_dialog(intended_action, actual_reality, story_context, history, pending_setup, goal, force_transition, location_constraint)
        return self.compare_and_decide_no_dialog(intended_action, actual_reality, story_context, history, pending_setup, goal, force_transition, location_constraint)
        

    def compare_and_decide_dialog(self, intended_action, actual_reality, story_context, history, pending_setup, goal=None, force_transition=False, location_constraint=None):
        history_text = "\n".join([f"- {a}" for a in history[-3:]]) if history else "First beat."
        
        setup_context = ""
        if pending_setup:
            setup_context = f"\nPREVIOUS SETUP: {pending_setup}\nThis was set up in the last beat and should now pay off or escalate."
        
        transition_directive = ""
        if force_transition:
            transition_directive = """
    CRITICAL: The character has walked away or turned their back. You MUST generate a "CUT TO:" that transitions to a NEW LOCATION or NEW CAMERA ANGLE where the character is clearly visible from the front or 3/4 view. Do NOT continue the current shot."""
        
        constraint_directive = ""
        if location_constraint:
            constraint_directive = f"\nCONSTRAINT: {location_constraint}"
        
        goal_directive = ""
        if goal:
            goal_directive = f"""
    NARRATIVE GOAL: {goal}

    CRITICAL: Every action you generate MUST move the characters closer to completing this goal. 
    - Evaluate what ACTUALLY happened in the video (ACTUAL SCENE STATE)
    - Choose the next action (physical OR verbal) that logically progresses toward the goal
    - If the characters deviated from the intended path, adapt and find a new route to the goal
    - The goal should be completed within 3-5 beats
    """
        
        if not history:
            task_directive = """TASK: This is the FIRST BEAT. 
    1. The ACTUAL SCENE STATE is the starting visual.
    2. Your NEXT_ACTION MUST be the specific physical action OR DIALOGUE described in the STORY CONTEXT. 
    3. Do not just advance the story; EXECUTE the story context as the immediate action. Characters can speak, tell jokes, or react verbally."""
        else:
            task_directive = """TASK: Apply "Yes, And..." improv logic with GOAL-DIRECTED PROGRESSION.
    1. YES: Accept the ACTUAL SCENE STATE as absolute truth (what actually happened, not what was intended).
    2. AND: Generate the next physical action OR DIALOGUE that moves toward the NARRATIVE GOAL.
    3. CRITICAL: This action must SET UP the next beat while progressing toward the goal. Include specific character dialogue if it serves the comedic or narrative goal (e.g., "The woman with red hair in a green shirt says...")."""

        prompt = f"""STORY CONTEXT: {story_context}
    {goal_directive}
    PREVIOUS INTENTION: {intended_action}
    ACTUAL SCENE STATE: {actual_reality}
    RECENT ACTIONS: {history_text}{setup_context}{transition_directive}{constraint_directive}

    {task_directive}

    Output format (STRICTLY follow this, no extra text):
    MATCH: [YES/PARTIAL/NO]
    ISSUES: [none, or specific problem]
    LOCATION: [brief location]
    CHARACTERS: [brief descriptions, including key visual identifiers like hair/clothing color]
    NEXT_ACTION: [1-2 sentences of story action. This CAN AND SHOULD include specific dialogue, comedic timing, or verbal reactions if it serves the goal (e.g., "The woman with red hair in a green shirt delivers a punchline while laughing").]
    CAMERA_FRAMING: [1 sentence of strict visual direction: lens, angle, lighting, movement]
    SETUP: [what this sets up for the next beat]
    GOAL_PROGRESS: [how this action moves toward completing the goal]
    """
        
        result = llm_analyze_media(
            media="", prompt=prompt,
            system="You are a film director and screenwriter specializing in comedic timing and character interaction. Every action (including dialogue, facial expressions, and physical comedy) must move toward the narrative goal while adapting to what actually happened. Use cinematic cuts to solve visibility issues.",
            max_tokens=300, temperature=0.7
        )['analysis']
        
        return result.strip()

    def compare_and_decide_no_dialog(self, intended_action, actual_reality, story_context, history, pending_setup, goal=None, force_transition=False, location_constraint=None):
        history_text = "\n".join([f"- {a}" for a in history[-3:]]) if history else "First beat."
        
        setup_context = ""
        if pending_setup:
            setup_context = f"\nPREVIOUS SETUP: {pending_setup}\nThis was set up in the last beat and should now pay off or escalate."
        
        transition_directive = ""
        if force_transition:
            transition_directive = """
    CRITICAL: The character has walked away or turned their back. You MUST generate a "CUT TO:" that transitions to a NEW LOCATION or NEW CAMERA ANGLE where the character is clearly visible from the front or 3/4 view. Do NOT continue the current shot."""
        
        constraint_directive = ""
        if location_constraint:
            constraint_directive = f"\nCONSTRAINT: {location_constraint}"
        
        goal_directive = ""
        if goal:
            goal_directive = f"""
    NARRATIVE GOAL: {goal}

    CRITICAL: Every action you generate MUST move the character closer to completing this goal. 
    - Evaluate what ACTUALLY happened in the video (ACTUAL SCENE STATE)
    - Choose the next action that logically progresses toward the goal
    - If the character deviated from the intended path, adapt and find a new route to the goal
    - The goal should be completed within 3-5 beats
    """
        
        if not history:
            task_directive = """TASK: This is the FIRST BEAT. 
    1. The ACTUAL SCENE STATE is the starting visual.
    2. Your NEXT_ACTION MUST be the specific physical action described in the STORY CONTEXT. 
    3. Do not just advance the story; EXECUTE the story context as the immediate action."""
        else:
            task_directive = """TASK: Apply "Yes, And..." improv logic with GOAL-DIRECTED PROGRESSION.
    1. YES: Accept the ACTUAL SCENE STATE as absolute truth (what actually happened, not what was intended).
    2. AND: Generate the next physical action that moves toward the NARRATIVE GOAL.
    3. CRITICAL: This action must SET UP the next beat while progressing toward the goal."""

        prompt = f"""STORY CONTEXT: {story_context}
    {goal_directive}
    PREVIOUS INTENTION: {intended_action}
    ACTUAL SCENE STATE: {actual_reality}
    RECENT ACTIONS: {history_text}{setup_context}{transition_directive}{constraint_directive}

    {task_directive}

    Output format (STRICTLY follow this, no extra text):
    MATCH: [YES/PARTIAL/NO]
    ISSUES: [none, or specific problem]
    LOCATION: [brief location]
    CHARACTERS: [brief descriptions]
    NEXT_ACTION: [1-2 sentences of pure story action that moves toward the goal]
    CAMERA_FRAMING: [1 sentence of strict visual direction: lens, angle, lighting, movement]
    SETUP: [what this sets up for the next beat]
    GOAL_PROGRESS: [how this action moves toward completing the goal]
    """
        
        result = llm_analyze_media(
            media="", prompt=prompt,
            system="You are a film director and screenwriter. Every action must move toward the narrative goal while adapting to what actually happened. Use cinematic cuts to solve visibility issues.",
            max_tokens=300, temperature=0.7
        )['analysis']
        
        return result.strip()

    def parse_decision(self, decision_text):
        lines = decision_text.split('\n')
        match = "UNKNOWN"
        issues = "none"
        location = ""
        characters = ""
        next_action = ""
        camera_framing = "static shot, medium framing"
        setup = ""
        goal_progress = ""
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith("MATCH:"):
                match = line.split(":", 1)[1].strip()
            elif line.upper().startswith("ISSUES:"):
                issues = line.split(":", 1)[1].strip()
            elif line.upper().startswith("LOCATION:"):
                location = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CHARACTERS:"):
                characters = line.split(":", 1)[1].strip()
            elif line.upper().startswith("NEXT_ACTION:"):
                next_action = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CAMERA_FRAMING:"):
                camera_framing = line.split(":", 1)[1].strip()
            elif line.upper().startswith("SETUP:"):
                setup = line.split(":", 1)[1].strip()
            elif line.upper().startswith("GOAL_PROGRESS:"):
                goal_progress = line.split(":", 1)[1].strip()
        
        return match, issues, location, characters, next_action, camera_framing, setup, goal_progress

    def _clean_analysis(self, raw_analysis):
        lines = raw_analysis.split('\n')
        clean_lines = []
        
        skip_keywords = ['analysis', 'discrepancies', 'issues', 'unexpected', 'summary', 'based on image', 'based on video']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(keyword in line.lower() for keyword in skip_keywords):
                continue
            clean_lines.append(line)
        
        return ' '.join(clean_lines)