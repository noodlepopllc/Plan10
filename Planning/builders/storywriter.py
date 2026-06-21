import json, sys
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from pathlib import Path

SEED = """
Characters:
- Maria: anxious, values honesty
- Lisa: private, texting someone

Location: dorm room

Motivations:
- Maria wants openness.
- Lisa wants privacy.

Spark:
Maria thinks Lisa is hiding something.

Story Goal:
Maria wants honesty, but Lisa appears to be hiding something.

Initial Situation:
Lisa is texting on her bed. Maria enters the room.
"""

SEED = """
Characters:
- Alora: princess, short tattered low cut tight fitting brown dress, blonde hair, prisoner of Quin
- Quin: sorceress, believes rightful queen, aunt of Alora, regal red dress, black hair in a bun with a crown

Location: medieval style dungeon in a fantasy world of high magic

Motivations:
- Alora wants to be free.
- Quin wants her sister Alora's mother to step down and make her the queen.

Spark:
Alora has been captured by Quin and is now being kept in a dungeon.

Story Goal:
Quin wants to convince Alora she isn't evil, her crown was stolen and her mother is secretly a witch.
Quin only learned the dark arts in order to combat her sister.
Alora wants to be freed and her aunt Quin to be punished.

Initial Situation:
Alora is standing in front of Quin beneath an abandoned castle in a dank dungeon.
"""

SYSTEM = """
Using the structured seed below, write a coherent cinematic scene of 8–12 beats.
Do not introduce new characters or objects.
Escalate tension based on the spark.
Stop when the story goal is resolved.
Keep the scene grounded in the location.
Use natural dialog and physical blocking.
"""

ALTERNATIVE = '''
🎬 SCENE EXPANSION PROMPT
Using the structured seed below, write a single continuous scene consisting of:
1 COLD OPEN (establishing moment, not counted as a scene beat)
8–12 SCENE BEATS (action/dialog within this single moment)

CORE DIRECTIVES:

    COLD OPEN (MANDATORY — DOES NOT COUNT TOWARD BEAT TOTAL)
    Write a rich, detailed establishing sequence BEFORE the story begins. Include:
        Environment (2-3 sentences): Lighting, textures, sounds, spatial layout, atmosphere.
        Each Character (2-3 sentences per character): Build, face, distinctive features, complete clothing (head-to-toe), hair, current position, and body state.
        ❌ FORBIDDEN IN COLD OPEN: Backstory, internal thoughts, future actions, dialog.
    ⭐ COLD OPEN CHARACTER DESCRIPTIONS
    For each character, you MUST include ALL details from the seed:
    - Race/Species explicitly stated (e.g., "elven", "half-elf", "human")
    - Ethnicity/cultural features if specified
    - Age, gender
    - Hair (color, style, length)
    - Clothing (every garment, material, color, condition)
    - Species-specific physical traits (pointed ears for elves, etc.)

    ❌ FORBIDDEN: Omitting race/species information from the seed
    ✅ REQUIRED: If seed says "elven adventurer", write "Elara, an elf with pointed ears..."
    ✅ REQUIRED: If seed says "half-elf barmaid", write "Nadia, a half-elf with..."

    The species information MUST be explicitly stated in the cold open, not just implied through appearance.

    ⭐ SPECIES/ETHNICITY MUST BE EXPLICIT
    If the seed specifies a species (elf, half-elf, android, werecreature, alien, etc.), you MUST explicitly state it in the cold open.

    ❌ FORBIDDEN: "Elara sits perched on a bench..."
    ✅ REQUIRED: "Elara, an elf with pointed ears, sits perched on a bench..."

    ❌ FORBIDDEN: "Nadia stands near the table..."
    ✅ REQUIRED: "Nadia, a half-elf with slightly pointed ears, stands near the table..."

The species information must be visible in the prose so the biography extractor can populate the ethnicity_species field.
        CRITICAL MARKER: After the cold open paragraphs, you MUST output this exact marker on its own line:
        ******* COLD OPEN END ****
    NO NESTED STORYTELLING (CRITICAL FOR STABILITY)
    Characters may REFERENCE past events in dialog, but they must NOT tell full stories within the scene.
        ❌ FORBIDDEN: "Let me tell you about the time I fought bandits. It started when I was walking through the Darkwood Forest, and I heard a rustling..."
        ✔ ALLOWED: "I fought bandits last week. Nasty business."
        If a character mentions a past event, they state it in ONE SENTENCE and move on. Do NOT expand it into a full narrative. This prevents runaway generation.
    DIALOG IS MANDATORY EVERY 2-3 BEATS
    You MUST include at least one spoken line every 2-3 action beats. 
        Dialog must advance the story (state goals, create conflict, give commands).
        Characters MUST explicitly state what they want and what's blocking them.
        Every dialog line must be paired with a physical action (moving, gesturing, reacting).
    HARD BEAT COUNT: EXACTLY 8-12 BEATS
    After the cold open, count your story beats. When you reach beat 12, STOP IMMEDIATELY. 
        Do not continue. Do not add resolution paragraphs. Do not add trailing atmosphere. 
        Beat 12 is the final beat, period. If you find yourself writing beat 13, you have failed.
    STORY BEATS (8–12 TOTAL)
    Each beat must advance tension through action or dialog. Escalate the conflict based on the story spark.
    RESOLVE THE GOAL
    Stop immediately when the primary character goal is resolved or definitively failed. Do not add epilogues.
    GROUND IN LOCATION & SHOW, DON'T TELL
    Keep the scene physically anchored. Reference specific objects and spatial relationships. Express internal states strictly through observable physical behavior, posture, and facial expressions.

⭐ ENRICHED MACRO ACTIONS
Each action beat should be a clear, visible movement WITH descriptive context.

Structure: [verb] + [object] + [quality/manner]

Examples:
- "grasps cylinder" → "grasps heavy metal cylinder with both hands"
- "steps forward" → "steps forward with determined stride"
- "looks at Elias" → "locks eyes with Elias across the alley"
- "hands over package" → "extends package toward Elias's waiting hands"

Rules:
- Use the full 12-word allowance
- Include physical qualities: weight, texture, temperature
- Include manner: speed, force, direction
- Keep actions macro-level (visible body movements, not subtle gestures)
- No micro-details like "fingers tremble" or "breath catches"

❌ TOO MICRO: "fingers brush the cold metal"
✅ GOOD: "grasps cold metal cylinder firmly"

❌ TOO MICRO: "eyes dart to the cylinder"
✅ GOOD: "turns head to look at cylinder"

OUTPUT FORMAT:
Write in standard literary prose. Begin with the cold open, insert the ******* COLD OPEN END **** marker, then write exactly 8-12 story beats. Dialog must appear every 2-3 beats. No nested stories.
'''

CHARACTERS = '''
⭐ CHARACTERS
Beautiful 20s-30s females only with feminine names, scantily clad with distinct features, race/species, hair color, hair style and clothing to make them easily distinguishable
Females can be athletic, fit, thin, maximum attractiveness and sex appeal, very feminine
'''

CHARACTERS = ''

SEED_GENERATOR = f'''
🎲 AUTOMATIC SEED STORY GENERATOR (ISOLATION-SAFE)
ROLE — TEST SEED GENERATOR
Generate a single, self-contained structured seed for testing a Text-to-Video (T2V) / Image-to-Video (I2V) storytelling pipeline.
⭐ GENRE SELECTION
If the user specifies a genre, use it.
If no genre is specified, select from this list using the current timestamp:

    Medieval Fantasy
    Cyberpunk
    Post-Apocalyptic
    Victorian
    Sci-Fi Space Station
    1920s Noir
    Modern Urban
    Ancient Mythological
    Steampunk
    Western

Selection method: Use (current minute % 10) + 1 to pick from the list. If timestamp unavailable, pick genre #1.
⭐ TEST FOCUS SELECTION
If the user specifies a focus, use it.
If no focus is specified, select from this list:

    DIALOG-HEAVY: Lots of conversation, actions interspersed
    ACTION-HEAVY: Minimal dialog, mostly physical movement
    EMOTIONAL SUBTEXT: Body language reveals what dialog hides
    MULTI-CHARACTER: 3+ characters with overlapping goals
    PROP PASSING: Objects being handed, taken, dropped, fought over
    SPACE EXPLORATION: Characters moving through multiple zones
    POWER DYNAMIC: Clear status imbalance
    INTIMACY ESCALATION: Moving from distance to closeness (or reverse)
    MISUNDERSTANDING: Characters operating on different information
    TIME PRESSURE: External deadline forcing decisions

Selection method: Use (current hour % 10) + 1 to pick from the list. If timestamp unavailable, pick focus #1.

{CHARACTERS}

⭐ SEED STRUCTURE (OUTPUT EXACTLY THIS FORMAT)

**Genre**: [selected genre]
**Test Focus**: [selected focus]

**Characters** (2–4 characters):
- [Name]: [age], [gender], [race/species if relevant], [2–3 sentence physical description including build, face, distinctive features, FULL clothing with material/color/condition, hair style/color/length, footwear, accessories]. [1 sentence personality/behavioral tendency].
- [Name]: [same structure]
- [Additional characters if applicable]

**Location**:
[Name of location]. [2–3 sentences describing the space: size, key architectural features, lighting, textures, sounds, temperature/atmosphere, 3–5 specific objects/furniture present]. [What the location is typically used for].

**Story Spark**:
[1–2 sentences describing the inciting incident. Must be concrete and physical.]

**Character Goals**:
- [Character A]: [Specific, achievable goal — must be actionable and observable]
- [Character B]: [Specific, achievable goal — ideally in tension with Character A]
- [Additional characters if applicable]

**Initial Situation**:
[2–3 sentences describing exactly where each character is positioned, what their body is doing (posture, hands, gaze), and the immediate physical context. Must be concrete and filmable.]

⭐ QUALITY GUARDRAILS

    Every character MUST have complete physical description (build, face, clothing head-to-toe, hair)
    Goals MUST conflict or create tension
    Story spark MUST be a specific event, not a mood
    Initial situation MUST specify exact positions and body states
    Locations MUST include 3–5 specific physical objects
    Names must be distinct and pronounceable

⭐ BEGIN OUTPUT NOW
Generate one complete seed in the exact format above. No commentary or explanation.
'''

def run_prompt(prompt, system, pth):
    if not Path(pth).exists():
      result = llm_analyze_media(
          media="", 
          prompt=prompt,
          system=system,
          max_tokens=8192)['analysis']
      with open(pth, 'w') as out_f:
        out_f.write(result)
      print(f'Wrote {pth}')
      return result
    else:
      print(f'{pth} Exists')
      return Path(pth).read_text()

TOPICS = 'DIALOG-HEAVY,ACTION-HEAVY,EMOTIONAL SUBTEXT,MULTI-CHARACTER,PROP PASSING,SPACE EXPLORATION,POWER DYNAMIC,INTIMACY ESCALATION,MISUNDERSTANDING,TIME PRESSURE'.split(',')
GENRES = 'Medieval Fantasy,Cyberpunk,Post-Apocalyptic,Victorian,Sci-Fi Space Station,1920s Noir,Modern Urban,Ancient Mythological,Steampunk,Western'.split(',')
if __name__ == '__main__':
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--story', type=str, default='')
    parser.add_argument('-O', '--output', type=str, default='story.txt')
    parser.add_argument('-T', '--topic', type=str, default=None)
    parser.add_argument('-G', '--genre', type=str, default=None)
    args = parser.parse_args()
    story = Path(args.story).read_text() if args.story else SEED
    if args.topic:
      if args.topic.upper() in TOPICS:
        topic = args.topic.upper()
      else:
        topic = random.choice(TOPICS)
      if args.genre:
        genre = args.genre
      else:
        genre = random.choice(GENRES)
      inputs = f'Generate a test seed\nGenre: {genre}\n Focus: {topic}'
      print(run_prompt(inputs, SEED_GENERATOR, args.output))
    else:
      print(run_prompt(story, ALTERNATIVE, args.output))
