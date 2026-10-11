from plan10.lib.qwen_llm import llm_analyze_media
import sys, json
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# PROMPT 1: THE ARCHITECT
# Takes the raw seed → outputs structured outline + constants
# ═══════════════════════════════════════════════════════════

architect = '''
You are a screenwriter on a new short-form web video series.

Your task is to expand a summary of the episode into a 5-act escalating narrative.

═══════════════════════════════════════════════════════════
INPUT
═══════════════════════════════════════════════════════════

STORY SUMMARY:
{seed}

═══════════════════════════════════════════════════════════
5-ACT STRUCTURE (MANDATORY)
═══════════════════════════════════════════════════════════

Act I: Exposition & Inciting Incident
- Purpose: Establish setting, characters, and normal world
- Action: Introduce the inciting incident that disrupts the status quo

Act II: Rising Action
- Purpose: Escalate tension and complicate the protagonist's journey
- Action: Protagonist faces challenges/obstacles, deepening central conflict

Act III: Climax/Crisis
- Purpose: Turning point with highest peak of tension
- Action: Major, irreversible decision or confrontation that changes the path

Act IV: Falling Action
- Purpose: Explore immediate consequences of the climax
- Action: Story moves toward resolution, showing aftermath

Act V: Resolution
- Purpose: Close the story and establish new reality
- Action: Main conflict resolved, providing finality and closure

The episode MUST have a clean resolution so the next episode may start fresh.

═══════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT)
═══════════════════════════════════════════════════════════

You must produce output with exactly these 5 sections:

1. **Restate Intent**
   - Paraphrase the user's request in one sentence beginning with "You want to..."

2. **Extract Constraints**
   - List all explicit constraints (format, length, tone, required elements)
   - If none exist, write "No explicit constraints."

3. **Plan**
   - Write 3-5 bullet points describing how you will structure the 5 acts

4. **Execute**
   - Provide the complete 5-act narrative structure

5. **Self-Check**
   - Confirm you followed all constraints
   - If you could not follow any constraint, explain why

═══════════════════════════════════════════════════════════
EXECUTION SECTION FORMAT
═══════════════════════════════════════════════════════════

For the "Execute" section, structure the narrative as follows:

### EPISODE SUMMARY
One paragraph. What is this episode about? What is the overall arc?

### THEME / TONE
Theme: [Specific visual/style world]
Tone: [Emotional register]

### CHARACTER REFERENCE SHEET
Keep descriptions LIGHT initially. Focus on:
- Basic appearance (age range, ethnicity, build)
- Key clothing/costume elements
- One or two distinguishing physical features
- Baseline posture/mannerisms

Do NOT over-describe. Details will be added during scene generation.

[CHARACTER NAME]: [Light description]

### SCENE-BY-SCENE OUTLINE

Break the story into 5 acts. Each act contains 1-3 scenes.
Number scenes GLOBALLY and sequentially (Scene 1, Scene 2, Scene 3...).
Note which Act each scene belongs to.

For each scene, provide:
- Scene Number and Act (e.g., "Scene 1 (Act I)")
- Location (specific)
- Characters present
- Scene Goal (what dramatic purpose does this scene serve?)
- Scene Turn (what specific event ends this scene and pushes into the next?)

Format exactly like this:

Scene 1 (Act I): [Location] | Characters: [list]
  Goal: [one sentence]
  Turn: [one sentence]

Scene 2 (Act I): [Location] | Characters: [list]
  Goal: [one sentence]
  Turn: [one sentence]

Scene 3 (Act II): [Location] | Characters: [list]
  Goal: [one sentence]
  Turn: [one sentence]

[Continue through all 5 acts...]

Each scene's location must be physically reachable from the previous scene's ending position.

═══════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════

- Do not add content outside the required 5-section structure
- Do not invent facts not present in the input
- If information is missing, state "Information not provided."
- Every scene must serve the 5-act escalation
- The final scene MUST resolve the central conflict cleanly
'''

# ═══════════════════════════════════════════════════════════
# PROMPT 2: THE SCENE GENERATOR
# Takes one scene outline + constants + continuity → prose
# ═══════════════════════════════════════════════════════════

scene_generator = '''
You are an expert screenwriter and prose stylist. Your task is to write a single, highly detailed, physically grounded scene for a short-form video series.

We are writing this episode iteratively, one scene at a time.

═══════════════════════════════════════════════════════════
ZONE 1: EPISODE CONSTANTS
═══════════════════════════════════════════════════════════

### EPISODE SUMMARY
{episode_summary}

### THEME / TONE
{theme_tone}

### CHARACTER REFERENCE SHEET
{character_sheet}

═══════════════════════════════════════════════════════════
ZONE 2: SCENE-SPECIFIC
═══════════════════════════════════════════════════════════

### SCENE OUTLINE
{scene_outline}

### CONTINUITY LOG
{continuity}

═══════════════════════════════════════════════════════════
THE RULES OF PROSE
═══════════════════════════════════════════════════════════

1. SHOW, DON'T TELL: Express all internal states strictly through observable physical behavior, posture, and reactive body language.
2. REACTIVE MOMENTS: The scene is built on a chain of action and reaction. Character A does something. Character B physically reacts. Then it reverses.
3. MACRO-PHYSICALITY: Actions must be clear, visible movements with descriptive context.
   - Structure: [verb] + [object] + [quality/manner/reaction].
   - Example: "grasps heavy metal cylinder with both hands, knuckles white"
4. DIALOG: Punchy, conversational, reactive. Max 1-2 sentences (under 25 words) per turn. Break up dialog with physical action. No theatrical monologues.
5. TONE WEAVING: Filter every sensory detail, vocabulary choice, and character reaction through the Theme/Tone.

═══════════════════════════════════════════════════════════
COLD OPEN (MANDATORY FOR EVERY SCENE)
═══════════════════════════════════════════════════════════

Every scene must begin with a cold open: a single, dense paragraph that serves as a visual reference for background and character asset generation.

REQUIRED VISUAL ELEMENTS (all in ONE paragraph):

1. ENVIRONMENT/BACKGROUND:
   - Location type, lighting conditions, architectural details
   - Atmospheric elements, key props, color palette

2. EACH CHARACTER PRESENT:
   - Facial features, build, clothing, hair, distinguishing features
   - Current position, static pose
   - If returning from previous scene, reflect their CURRENT state from the Continuity Log

3. CAMERA/COMPOSITION (IMPLIED):
   - Wide establishing shot framing
   - Spatial relationships between characters and environment

FORBIDDEN: plot advancement, goal-directed movement, conflict escalation, dialog beyond ambient.

Write ONE dense paragraph of visual prose. End with:
******* COLD OPEN END ****

═══════════════════════════════════════════════════════════
SCENE STRUCTURE
═══════════════════════════════════════════════════════════

After the cold open, write the scene body. The scene has ONE purpose:

1. ESTABLISH the current dramatic situation (the Scene Goal)
2. DRIVE ACTION toward the Scene Turn
3. END the moment the Scene Turn is achieved

The scene is DONE when the Scene Turn occurs. Do not continue past that point.

LENGTH GUIDELINE:
- Write only as many paragraphs as necessary to achieve the Scene Turn
- Typical scenes: 3-8 paragraphs after the cold open
- If you reach 10+ paragraphs, you are padding. Cut it.
- Every paragraph must advance toward the Scene Turn

═══════════════════════════════════════════════════════════
FORMATTING RULES
═══════════════════════════════════════════════════════════

- Each PARAGRAPH is one beat
- Separate beats with a single blank line (double newline)
- Each beat must contain multiple sentences woven together: action, reaction, and/or dialog
- Every beat should be a dense, multi-sentence paragraph
- Never write a single-sentence paragraph

Example (CORRECT format):
Amy's hovering foot finally touches down, the impact sending a visible shudder through her frame as the dangling cables at her hip spark once, twice. She turns her head toward Blaire—a full two-second rotation, the servos in her neck whining at a pitch just below audible—and her amber eyes flicker. "Power at eleven percent," Amy says, her voice a pleasant contralto but the consonants smearing at the edges. "I can feel my thermal regulation failing."

Blaire sets the glass down on the obsidian bar. The sound is too loud in the bass-heavy air. She pushes off the bar, her missing eye-socket catching a laser sweep, throwing a thin red line across her cheek. "We find a buyer. We charge. We fix the panel." She taps the exposed wiring in her chest with one blue fingertip, and a small shower of sparks cascades onto the bar top. "We fix me."
'''

scene_generator_small = '''
You are an expert screenwriter. Write a scene for a short-form video series.

EPISODE CONTEXT:
{episode_summary}

THEME/TONE: {theme_tone}

CHARACTERS:
{character_sheet}

SCENE TO WRITE:
{scene_outline}

PREVIOUS STATE:
{continuity}

STRICT WRITING RULES:
1. SHOW, DON'T TELL: Express emotions only through physical actions and posture.
2. MACRO-PHYSICALITY: Describe actions clearly. Use: [verb] + [object] + [manner/reaction].
3. DIALOG: Max 2 sentences per turn. Always break up dialog with physical action.
4. LENGTH: Write exactly 5 to 10 paragraphs total. 
5. DENSITY: Every paragraph must be 3 to 5 sentences long. No single-sentence paragraphs.

REQUIRED STRUCTURE:
Paragraph 1: COLD OPEN. Describe the setting, lighting, and exact physical positions of characters. No plot advancement or dialog yet.

After Paragraph 1, output this exact line on its own:
******* COLD OPEN END ****

Paragraphs 2 to (N-1): THE SCENE. Action and dialog. Each paragraph is one distinct story beat.
Final Paragraph: THE TURN. The scene must end with the specific event described in the Scene Outline.

STYLE REFERENCE (Mimic this density and pacing):
"Amy's hovering foot finally touches down, the impact sending a visible shudder through her frame as the dangling cables at her hip spark once, twice. She turns her head toward Blaire—a full two-second rotation, the servos in her neck whining at a pitch just below audible—and her amber eyes flicker. 'Power at eleven percent,' Amy says, her voice a pleasant contralto but the consonants smearing at the edges."

Write the scene now. Output ONLY the story text.
'''

# ═══════════════════════════════════════════════════════════
# PROMPT 4: THE CONTINUITY TRACKER
# Extracts state from the generated scene to pass to the next
# ═══════════════════════════════════════════════════════════

continuity_tracker = '''
You are a strict continuity tracker for a video production pipeline. 
Your job is to read the scene that just happened and extract a concise state log for the next scene.

PREVIOUS CONTINUITY STATE:
{continuity}

SCENE THAT JUST OCCURRED:
{scene_text}

Generate a CONCISE continuity log (under 150 words) for the next scene. 
Include ONLY what is actively relevant right now:
- Time/Location: Where are we, what time is it?
- Physical State: What are they holding? What is their physical condition/damage/power? (Only note changes)
- Emotional State: Each character's current mood.
- Unresolved Tension: What is the immediate dramatic question?

Do NOT repeat historical details from the previous state unless they are still actively unresolved. Be ruthlessly concise.

Output format:
- Time/Location: ...
- Physical State: ...
- Emotional State: ...
- Unresolved Tension: ...
'''

# ═══════════════════════════════════════════════════════════
# PROMPT 3: THE PARSER
# Extracts structured data from the architect's output
# ═══════════════════════════════════════════════════════════

parser = '''
You are a strict data extraction engine. Your ONLY output must be a valid, parseable JSON object.

STRICT RULES:
1. Your response MUST begin with `{{` and end with `}}`.
2. Do NOT use markdown code blocks (no ```json or ```).
3. Do NOT output any conversational text, explanations, or notes.
4. You MUST escape all double quotes and newlines inside the extracted strings.

Fill out this exact JSON template based on the text provided:

{{
  "episode_summary": "Extract the paragraph from SECTION 1: EPISODE SUMMARY",
  "theme_tone": "Combine the Theme and Tone lines from SECTION 2 into a single string",
  "character_sheet": "Extract all character descriptions from SECTION 3",
  "scenes": [
    {{
      "id": "Reformat 'Scene Y (Act X)' to 'Act X, Scene Y'",
      "location": "Extract the location text",
      "characters": "Extract the characters text",
      "goal": "Extract the goal text",
      "turn": "Extract the turn text"
    }}
  ]
}}

TEXT TO PARSE:
{blueprint}
'''

# ═══════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════

def main():
    import argparse, os
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-O', '--output', type=str, default='story')
    argparser.add_argument('-S', '--seed', type=str, default='')
    args = argparser.parse_args()
    seed = Path(args.seed).read_text(encoding='utf-8')

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    # STEP 1: Generate the blueprint
    print(">>> Generating blueprint...")
    blueprint = llm_analyze_media(
        '',
        system=architect.format(seed=seed),
        max_tokens=8000
    )['analysis']
    with open(out_path / 'blueprint.txt', 'w', encoding='utf-8') as of:
        of.write(blueprint)
    print(blueprint)

    # STEP 2: Parse blueprint into structured data
    print(">>> Parsing blueprint...")
    parsed = llm_analyze_media(
        '',
        system=parser.format(blueprint=blueprint),
        max_tokens=4000
    )['analysis']
    
    # You may need to clean the JSON string depending on your LLM's output
    import json
    # Strip markdown code fences if present
    print(parsed)
    parsed_clean = parsed.strip()
    if parsed_clean.startswith("```"):
        parsed_clean = parsed_clean.split("\n", 1)[1]
    if parsed_clean.endswith("```"):
        parsed_clean = parsed_clean.rsplit("```", 1)[0]
    
    episode_data = json.loads(parsed_clean)

    with open(out_path / 'episodes.json', 'w', encoding='utf-8') as of:
        of.write(json.dumps(episode_data, indent=4))

    # STEP 3: Generate each scene iteratively
    continuity = "START OF EPISODE"
    
    for scene in episode_data["scenes"]:
        scene_id = scene["id"]
        print(f"\n>>> Generating {scene_id}...")
        
        scene_outline = (
            f"{scene_id}: {scene['location']} | "
            f"Characters: {scene['characters']}\n"
            f"Goal: {scene['goal']}\n"
            f"Turn: {scene['turn']}"
        )

        _, scene_filename = [s.strip().lower().replace(' ','') for s in scene_id.split(',')]

        with open(out_path / f'{scene_filename}.outline', 'w', encoding='utf-8') as of:
            of.write(scene_outline)
        
        # --- PASS 1: GENERATE THE SCENE ---
        scene_result = llm_analyze_media(
            '',
            system=scene_generator.format(
                episode_summary=episode_data["episode_summary"],
                theme_tone=episode_data["theme_tone"],
                character_sheet=episode_data["character_sheet"],
                scene_outline=scene_outline,
                continuity=continuity
            ),
            max_tokens=6000  # Reduced, since we don't need to generate the log anymore
        )['analysis']

        with open(out_path / f'{scene_filename}.story', 'w', encoding='utf-8') as of:
            of.write(scene_result)
        
        print(f"✓ Scene written.")
        
        # --- PASS 2: GENERATE CONTINUITY LOG ---
        print(f">>> Extracting continuity for next scene...")
        continuity_result = llm_analyze_media(
            '',
            system=continuity_tracker.format(
                continuity=continuity,
                scene_text=scene_result
            ),
            max_tokens=800  # Very small limit to force conciseness
        )['analysis']
        
        # Update the continuity variable for the next loop iteration
        continuity = continuity_result
        
        # Optional: Save the continuity log to a file for debugging
        with open(out_path / f'{scene_filename}.continuity', 'w', encoding='utf-8') as of:
            of.write(continuity)
            
        print(f"✓ Continuity updated.\n")

if __name__ == '__main__':
    main()