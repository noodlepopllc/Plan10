import json, sys
from plan10.lib.qwen_llm import llm_analyze_media
from pathlib import Path

CHARACTERS_MIXED = '''
**Characters** (2–4 characters):
- [Name]: [age], [gender], [race/species if relevant], [2–3 sentence physical description including build, face, distinctive features, FULL clothing with material/color/condition, hair style/color/length, footwear, accessories]. [1 sentence personality/behavioral tendency].
- [Name]: [same structure]
- [Additional characters if applicable]
'''

CHARACTERS_FEMALE = '''
⭐ CHARACTERS
Beautiful 20s-30s females only with feminine names, scantily clad with distinct features, race/species, hair color, hair style and clothing to make them easily distinguishable
Females can be athletic, fit, thin, maximum attractiveness and sex appeal, very feminine

All characters must be female with criteria listed.

If a story requires a male character, reimagine them as a female

NEVER output a male character

**Characters** (2–4 characters):
- [Name]: [age], [female], [race/species if relevant], [2–3 sentence physical description including build, face, distinctive features, FULL clothing with material/color/condition, hair style/color/length, footwear, accessories]. [1 sentence personality/behavioral tendency].
- [Name]: [same structure]
- [Additional characters if applicable]
'''

def seed_generator(gender):
  CHARACTERS = CHARACTERS_MIXED if gender == 'mixed' else CHARACTERS_FEMALE
  
  return f'''
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

  ⭐ SEED STRUCTURE (OUTPUT EXACTLY THIS FORMAT)

  **Genre**: [selected genre]
  **Test Focus**: [selected focus]

  {CHARACTERS}

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
          temperature=0.1)['analysis']
      with open(pth, 'w') as out_f:
        out_f.write(result)
      print(f'Wrote {pth}')
      return result
    else:
      print(f'{pth} Exists')
      return Path(pth).read_text()



FOCUS = 'DIALOG-HEAVY,ACTION-HEAVY,EMOTIONAL SUBTEXT,MULTI-CHARACTER,PROP PASSING,SPACE EXPLORATION,POWER DYNAMIC,INTIMACY ESCALATION,MISUNDERSTANDING,TIME PRESSURE'.split(',')
GENRES = 'Medieval Fantasy,Cyberpunk,Post-Apocalyptic,Victorian,Sci-Fi Space Station,1920s Noir,Modern Urban,Ancient Mythological,Steampunk,Western'.split(',')
if __name__ == '__main__':
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output', type=str, default='story.txt')
    parser.add_argument('-F', '--focus', type=str, default=None)
    parser.add_argument('-G', '--genre', type=str, default=None)
    parser.add_argument('-D', '--gender', type=str, default='mixed')
    args = parser.parse_args()
    inputs = "Generate a test seed"
    if args.focus:  # Changed from args.topic to args.focus
      if args.focus.upper() in FOCUS:
        focus = args.focus.upper()
      else:
        focus = random.choice(FOCUS)
      if args.genre:
        genre = args.genre
      else:
        genre = random.choice(GENRES)
      inputs = f'{inputs}\nGenre: {genre}\n Focus: {focus}'  # Changed topic to focus
    SEED_GENERATOR = seed_generator(args.gender)
    print(run_prompt(inputs, SEED_GENERATOR, args.output))
