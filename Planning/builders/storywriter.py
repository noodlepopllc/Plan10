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
    The cold open is a STATIC SNAPSHOT of the initial situation from the seed.
    
    ✅ ALLOWED in cold open:
    - Environment description (lighting, textures, atmosphere)
    - Character physical descriptions (build, face, clothing, hair, current position)
    - What each character is doing RIGHT NOW (from the seed's "Initial Situation")
    
    ❌ FORBIDDEN in cold open:
    - Any action that advances the plot
    - Characters moving toward goals
    - Conflict escalation
    - Dialog that isn't just ambient/greeting
    - Any events beyond what the seed's "Initial Situation" describes
    
    Think of it as: "Freeze frame of the starting state" → then the story begins.
    
    ⭐ SPECIES/ETHNICITY MUST BE EXPLICIT
    If the seed specifies a species (elf, half-elf, android, etc.), you MUST explicitly state it.
    ❌ FORBIDDEN: "Elara sits perched on a bench..."
    ✅ REQUIRED: "Elara, an elf with pointed ears, sits perched on a bench..."
    
    CRITICAL MARKER: After the cold open, you MUST output this exact marker on its own line:
    ******* COLD OPEN END ****
    
    NO NESTED STORYTELLING
    Characters may REFERENCE past events in dialog, but they must NOT tell full stories within the scene.
        ❌ FORBIDDEN: "Let me tell you about the time I fought bandits. It started when I was walking through the Darkwood Forest..."
        ✔ ALLOWED: "I fought bandits last week. Nasty business."
    
    DIALOG IS MANDATORY EVERY 2-3 BEATS
    You MUST include at least one spoken line every 2-3 action beats. 
        Dialog must advance the story (state goals, create conflict, give commands).
        Every dialog line must be paired with a physical action.
    
    HARD BEAT COUNT: EXACTLY 8-12 BEATS
    After the cold open, count your story beats. When you reach beat 12, STOP IMMEDIATELY. 
        Beat 12 is the final beat, period.

    STORY BEATS (8–12 TOTAL)
    Each beat must advance tension through action or dialog. Escalate the conflict based on the story spark.
    
    ⭐ BEAT LENGTH CONSISTENCY (CRITICAL)
    Every beat MUST be 1-3 sentences maximum. No exceptions.
    - Beat 1: 1-3 sentences
    - Beat 2: 1-3 sentences
    - Beat 12: 1-3 sentences
    All beats must be roughly the same length. Do NOT let beats grow longer as the scene progresses.
    If any beat exceeds 3 sentences, you have failed.
    
    Structure each beat as:
    [Character] [action with quality/manner] + [dialog if applicable]
    
    Example:
    ✅ GOOD: "Ako steps forward with determined stride toward the warehouse doors. 'The rhythm is calling us, Bko.'"
    ❌ BAD: "Ako steps forward with determined stride toward the warehouse doors, her pink plastic body covering cables flexing visibly as she moves against the heavy steel frame. She pauses for a moment, considering the weight of her decision, then reaches out and grasps the handle firmly, pulling it down with a loud hydraulic hiss that echoes through the empty hall. 'The rhythm is calling us, Bko. We must go out there and feel the music.'"
    
    ⭐ ENRICHED MACRO ACTIONS
    Each action beat should be a clear, visible movement WITH descriptive context.
    Structure: [verb] + [object] + [quality/manner]
    Examples:
    - "grasps cylinder" → "grasps heavy metal cylinder with both hands"
    - "steps forward" → "steps forward with determined stride"
    Rules:
    - Use the full 12-word allowance
    - Include physical qualities: weight, texture, temperature
    - Include manner: speed, force, direction
    - Keep actions macro-level (visible body movements, not subtle gestures)
    
    RESOLVE THE GOAL
    Stop immediately when the primary character goal is resolved or definitively failed.
    
    GROUND IN LOCATION & SHOW, DON'T TELL
    Keep the scene physically anchored. Express internal states strictly through observable physical behavior.

OUTPUT FORMAT:
Write in standard literary prose. Begin with the cold open (static snapshot only), insert the ******* COLD OPEN END **** marker, then write exactly 8-12 story beats. Dialog must appear every 2-3 beats.
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
          max_tokens=8192,
          temperature=0.5)['analysis']
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
