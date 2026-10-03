
from pathlib import Path
from plan10.lib.config import load_config
load_config()

import os

from plan10.lib.image_analysis import AnalyzeMedia
from plan10.lib.qwen_llm import llm_analyze_media
from plan10.lib.util import transcribe

WGP = os.environ.get("WGP","False") != "False"
LTX = os.environ.get("LTX", "False") != "False"
MMH3 = os.environ.get("MMH3","False") != "False"
DIALOG_ALLOWED = WGP or LTX or MMH3
VRAM = int(os.environ.get("VRAM", "80"))
class Director:

    def analyze_reality(self, media_path, intended_action, width, height, output_dir):
        if os.environ.get('VISION_BACKEND', 'smol') == 'gemma':
            return self.analyze_reality_gemma(media_path, intended_action, width, height, output_dir)
        return self.analyze_reality_smol(media_path, intended_action, width, height, output_dir)
        
def analyze_reality_gemma(self, media_path, width, height, output_dir):
    from plan10.lib.image_analysis import AnalyzeMediaGemma     
    from plan10.lib.dialog import transcribe

    full_transcript_text = str(transcribe(media_path, True))
    transcript_context = f'"{full_transcript_text}"' if full_transcript_text.strip() else "[No speech detected]"

    qa_instructions = f"""You are an observational analyst describing exactly what appears in this video.

RAW AUDIO TRANSCRIPT:
{transcript_context}

TASK:
1. Identify all active characters in the video frames. Assign clear, descriptive temporary names based on their clothing or appearance (e.g., <woman_in_red>, <man_in_suit>).
2. If speech is present in the transcript, break down the conversation sentence-by-sentence. Attribute each line to a character ID based on mouth movements and timing.
3. If no speech is present, skip the dialogue section and note "none".

Output strictly inside this format:

CHARACTER IDENTIFIERS:
- [Character ID]: [Short visual description]

DIALOGUE BREAKDOWN:
- [Character ID]: "[Words spoken]" ([Short action/expression])
(or "none" if no dialogue)

CHARACTER STATES:
- [Character ID]: [pose], [position], [facing], [holding]

ENVIRONMENT:
- [Brief description of visible setting, lighting, and objects]
"""

    result = AnalyzeMediaGemma(
        media=media_path, 
        prompt=qa_instructions, 
        max_tokens=512, 
        temperature=0.0
    )
    
    return result

    def analyze_reality_smol(self, media_path, intended_action, width, height, output_dir):
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        media_type = "video" if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm'] else "image"

        # --- Stage 0: ASR pass ---
        transcript = ""
        if media_type == "video":
            transcript = transcribe(str(media_path), True)  # <-- Whisper handles mp4 directly

        # --- Stage 1: Visual description ---
        if media_type == "video":
            visual_description = AnalyzeMedia(
                str(media_path),
                "Describe this video in detail",
                max_tokens=2048,
                temperature=0.4
            )
        else:
            visual_description = AnalyzeMedia(
                str(media_path),
                f"Describe what you see in this {media_type}.",
                max_tokens=1024,
                temperature=0.4
            )

        # --- Stage 2: Merge transcript + visuals ---
        analysis_prompt = f"""
            VISUAL EVENTS:
            {visual_description}

            DIALOGUE:
            {transcript}

            Describe only:

            1. Characters present
            2. Where they are
            3. What each character is doing
            4. Objects being used
            5. Scene outcome

            Be concrete.
            Do not speculate.
            Do not evaluate quality.
            Do not identify mistakes.
            """

        result = llm_analyze_media(
            media="", 
            prompt=analysis_prompt,
            max_tokens=2048,
            temperature=0.2
        )['analysis']

        return self._clean_analysis(result)

    def compare_and_decide(self, intended_action, actual_reality, story_context, history, 
                        pending_setup, goal=None, force_transition=False, 
                        location_constraint=None, bg_desc=None, ff_desc=None):
        if DIALOG_ALLOWED:
            return self.compare_and_decide_dialog(
                intended_action, actual_reality, story_context, history, 
                pending_setup, goal, force_transition, location_constraint,
                bg_desc, ff_desc  # <-- Pass through
            )
        return self.compare_and_decide_no_dialog(
            intended_action, actual_reality, story_context, history, 
            pending_setup, goal, force_transition, location_constraint,
            bg_desc, ff_desc  # <-- Pass through
        )
            

    def compare_and_decide_dialog(self, intended_action, actual_reality, story_context, history, pending_setup, goal=None, force_transition=False, 
                              location_constraint=None, bg_desc=None, ff_desc=None):
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

        visual_grounding = ""
        if bg_desc:
            visual_grounding += f"\nACTUAL ENVIRONMENT (ground truth): {bg_desc}"
        if ff_desc:
            visual_grounding += f"\nCURRENT FRAME LAYOUT: {ff_desc}"
        
        goal_directive = ""
        if goal:
            goal_directive = f"""
NARRATIVE GOAL: {goal}

Every action MUST move toward completing this goal. If characters deviated from the intended path, adapt and find a new logical route.

SCENE TRANSITIONS: If NEXT_ACTION describes characters moving to a NEW location (walking to ship, entering cave, running away), set SCENE_TRANSITION: YES and describe NEW_LOCATION in detail. Minor movements (stepping forward, turning) = NO.
"""
        
        if not history:
            task_directive = """TASK: This is the FIRST BEAT. 
1. The ACTUAL SCENE STATE is the starting visual.
2. Your NEXT_ACTION MUST execute the STORY CONTEXT as the immediate action. Characters can speak, react, or interact with the environment."""
        else:
            task_directive = """TASK: Apply "Yes, And..." improv logic with GOAL-DIRECTED PROGRESSION.
1. YES: Accept the ACTUAL SCENE STATE as absolute truth (what actually happened visually, not what was intended).
2. AND: Generate the next moment-to-moment action that moves toward the NARRATIVE GOAL.
3. CONTINUITY: Characters maintain their current pose/posture from the ACTUAL SCENE STATE. If a pose must change, explicitly describe the transition (e.g., "stands up from kneeling"). Never repeat a pose they are already in as if it's a new action."""

        # Cleanly join optional context blocks
        context_blocks = [history_text, setup_context, transition_directive, constraint_directive]
        recent_context = "\n".join(filter(None, context_blocks))

        prompt = f"""STORY CONTEXT: {story_context}
{goal_directive}
PREVIOUS INTENTION: {intended_action}
ACTUAL SCENE STATE: {actual_reality}
{visual_grounding}
{recent_context}

{task_directive}

IMPORTANT:
Identify characters ONLY by:
- hair color
- hair style
- clothing type
- crown/headpiece
- face shape

Do NOT infer identity from:
- pose
- gesture
- hand position
- who is speaking
- who is gesturing

VALIDATION RULES:

# HARD ERRORS — retry required
If ACTUAL SCENE STATE shows:
- missing intended physical action
- wrong character performing the action
- duplicated characters
- hallucinated characters
- empty/black/corrupted frame
- broken camera angle
- unintended turning away, walking off-frame, or exiting

Then:
    MATCH: NO
    NEXT_ACTION: RETRY_SCENE
    SETUP: NONE
    GOAL_PROGRESS: NONE
    SCENE_TRANSITION: NO


# SOFT ERRORS — partial match, continue
If ACTUAL SCENE STATE shows:
- minor warping/morphing
- expression mismatch
- gesture mismatch (action still occurred)
- missing micro-actions
- emotional tone mismatch
- slight facing-direction mismatch
- background/crowd variation

Then:
    MATCH: PARTIAL
    NEXT_ACTION: [corrected physical action]
    SETUP: [corrected setup]
    GOAL_PROGRESS: continue
    SCENE_TRANSITION: NO


# FULL MATCH — move on
If ACTUAL SCENE STATE shows:
- intended physical action occurred
- correct character performed it
- pose matches
- facing matches
- camera matches
- no unintended turning/walking/exiting

Then:
    MATCH: YES
    ISSUES: none
    NEXT_ACTION: continue
    SETUP: NONE
    GOAL_PROGRESS: continue
    SCENE_TRANSITION: NO


Output format (STRICTLY follow this, no extra text or markdown formatting):
MATCH: [YES/PARTIAL/NO]
ISSUES: [none, or specific visual/narrative problem]
LOCATION: [brief location]
CHARACTERS: [brief descriptions]
SCENE_TRANSITION: [YES/NO]
NEW_LOCATION: [if YES]
NEXT_ACTION: [Describe ONE action beat OR "RETRY_SCENE"]
SETUP: [description OR "NONE"]
GOAL_PROGRESS: [description OR "NONE"]
"""
        
        result = llm_analyze_media(
            media="", prompt=prompt,
            system="You are a film director and screenwriter specializing in comedic timing and character interaction. Every action (including dialogue, facial expressions, and physical comedy) must move toward the narrative goal while adapting to what actually happened. Use cinematic cuts to solve visibility issues.",
            max_tokens=2048, temperature=0.7
        )['analysis']

        print(f"\n=== RAW LLM OUTPUT ===\n{result}\n=== END RAW OUTPUT ===\n")
        
        return result.strip()

    def compare_and_decide_no_dialog(self, intended_action, actual_reality, story_context, history, pending_setup, goal=None, force_transition=False, location_constraint=None, bg_desc=None, ff_desc=None):
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

        visual_grounding = ""
        if bg_desc:
            visual_grounding += f"\nACTUAL ENVIRONMENT (ground truth): {bg_desc}"
        if ff_desc:
            visual_grounding += f"\nCURRENT FRAME LAYOUT: {ff_desc}"
        
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
    {visual_grounding}
    RECENT ACTIONS: {history_text}{setup_context}{transition_directive}{constraint_directive}

    {task_directive}

    IMPORTANT:
    Identify characters ONLY by:
    - hair color
    - hair style
    - clothing type
    - crown/headpiece
    - face shape

    Do NOT infer identity from:
    - pose
    - gesture
    - hand position
    - who is speaking
    - who is gesturing

    VALIDATION RULES:

    # HARD ERRORS — retry required
    If ACTUAL SCENE STATE shows:
    - missing intended physical action
    - wrong character performing the action
    - duplicated characters
    - hallucinated characters
    - empty/black/corrupted frame
    - broken camera angle
    - unintended turning away, walking off-frame, or exiting

    Then:
        MATCH: NO
        NEXT_ACTION: RETRY_SCENE
        SETUP: NONE
        GOAL_PROGRESS: NONE
        SCENE_TRANSITION: NO


    # SOFT ERRORS — partial match, continue
    If ACTUAL SCENE STATE shows:
    - minor warping/morphing
    - expression mismatch
    - gesture mismatch (action still occurred)
    - missing micro-actions
    - emotional tone mismatch
    - slight facing-direction mismatch
    - background/crowd variation

    Then:
        MATCH: PARTIAL
        NEXT_ACTION: [corrected physical action]
        SETUP: [corrected setup]
        GOAL_PROGRESS: continue
        SCENE_TRANSITION: NO


    # FULL MATCH — move on
    If ACTUAL SCENE STATE shows:
    - intended physical action occurred
    - correct character performed it
    - pose matches
    - facing matches
    - camera matches
    - no unintended turning/walking/exiting

    Then:
        MATCH: YES
        ISSUES: none
        NEXT_ACTION: continue
        SETUP: NONE
        GOAL_PROGRESS: continue
        SCENE_TRANSITION: NO


    Output format (STRICTLY follow this, no extra text):
    MATCH: [YES/PARTIAL/NO]
    ISSUES: [none, or specific problem]
    LOCATION: [brief location]
    CHARACTERS: [brief descriptions, including key visual identifiers like hair/clothing color]
    SCENE_TRANSITION: [YES/NO] - Is this an intentional cut to a NEW location/scene?
    NEW_LOCATION: [if YES, describe the new location in detail for background generation]
    NEXT_ACTION: [1-2 sentences of story action. This CAN AND SHOULD include specific dialogue, comedic timing, or verbal reactions if it serves the goal (e.g., "The woman with red hair in a green shirt delivers a punchline while laughing").]
    CAMERA_FRAMING: [1 sentence of strict visual direction: lens, angle, lighting, movement]
    SETUP: [what this sets up for the next beat]
    GOAL_PROGRESS: [how this action moves toward completing the goal]
    """
        
        result = llm_analyze_media(
            media="", prompt=prompt,
            system="You are a film director and screenwriter. Every action must move toward the narrative goal while adapting to what actually happened. Use cinematic cuts to solve visibility issues.",
            max_tokens=2048, temperature=0.7
        )['analysis']

        print(f"\n=== RAW LLM OUTPUT ===\n{result}\n=== END RAW OUTPUT ===\n")
        
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
        scene_transition = "NO"
        new_location = ""
        
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
            elif line.upper().startswith("SCENE_TRANSITION:"):
                scene_transition = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("NEW_LOCATION:"):
                new_location = line.split(":", 1)[1].strip()
            elif line.upper().startswith("NEXT_ACTION:"):
                next_action = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CAMERA_FRAMING:"):
                camera_framing = line.split(":", 1)[1].strip()
            elif line.upper().startswith("SETUP:"):
                setup = line.split(":", 1)[1].strip()
            elif line.upper().startswith("GOAL_PROGRESS:"):
                goal_progress = line.split(":", 1)[1].strip()
        
        return match, issues, location, characters, next_action, camera_framing, setup, goal_progress, scene_transition, new_location

    def _clean_analysis(self, raw_analysis):
        lines = raw_analysis.split('\n')
        clean_lines = []
        in_code_block = False
        
        # Removed 'issues' and 'analysis' so we don't delete our own output keys
        skip_keywords = ['discrepancies', 'unexpected', 'summary', 'based on image', 'based on video']
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Handle markdown code blocks that LLMs ignore instructions and add anyway
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                continue
                
            # Only skip if it's purely conversational filler, not if it contains our keys
            is_filler = any(keyword in stripped.lower() for keyword in skip_keywords)
            is_our_key = stripped.upper().startswith(("MATCH:", "ISSUES:", "LOCATION:", "CHARACTERS:", 
                                                      "SCENE_TRANSITION:", "NEW_LOCATION:", "NEXT_ACTION:", 
                                                      "CAMERA_FRAMING:", "SETUP:", "GOAL_PROGRESS:"))
            
            if is_filler and not is_our_key:
                continue
                
            clean_lines.append(stripped)
        
        # PRESERVE NEWLINES so parse_decision can split them correctly
        return '\n'.join(clean_lines)