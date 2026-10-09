import json, sys, random
sys.stdout.reconfigure(encoding='utf-8')

from plan10.lib.qwen_llm import llm_analyze_media
from pathlib import Path

CHARACTERS_MIXED = '''
**Characters** (2–4 characters):
- [First Name]: [age], [gender], [race/species if relevant], [2–3 sentence physical description including build, face, distinctive features, FULL clothing with material/color/condition, hair style/color/length, footwear, accessories]. [1 sentence personality/behavioral tendency].
- [First Name]: [same structure]
- [Additional characters if applicable]
'''

CHARACTERS_FEMALE = '''
⭐ CHARACTERS
Beautiful 20s-30s females only with feminine names, scantily clad with distinct features, race/species, hair color, hair style and clothing to make them easily distinguishable
Females can be athletic, fit, thin, maximum attractiveness and sex appeal, very feminine

All characters must be female with criteria listed.

If a story requires a male character, reimagine them as a female

NEVER output a male character

**Characters** ({char_count}, no more, no less):
- [First Name - feminine only, no ambiguous names]: [age], [female], [race/species if relevant], [2–3 sentence physical description including build, face, distinctive features, FULL clothing with material/color/condition, hair style/color/length, footwear, accessories, accentuate female features in face chest, waist and body]. [1 sentence personality/behavioral tendency].
- [First Name]: [same structure]
- [Additional characters if applicable]
'''

def seed_generator(gender, genre, focus, char_count):
  CHARACTERS = CHARACTERS_MIXED if gender == 'mixed' else CHARACTERS_FEMALE
  
  return f'''
  🎲 AUTOMATIC SEED STORY GENERATOR (ISOLATION-SAFE)
  ROLE — TEST SEED GENERATOR
  Generate a single, self-contained structured seed for testing a Text-to-Video (T2V) / Image-to-Video (I2V) storytelling pipeline.
  
  ⭐ GENRE: {genre}
  ⭐ TEST FOCUS: {focus}

  ⭐ SEED STRUCTURE (OUTPUT EXACTLY THIS FORMAT)

  **Genre**: {genre}
  **Test Focus**: {focus}

  {CHARACTERS.format(char_count=char_count)}

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
      You MUST output {char_count}. Not one more, not one fewer.

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
      with open(pth, 'w', encoding='utf-8') as out_f:
        out_f.write(result)
      print(f'Wrote {pth}')
      return result
    else:
      print(f'{pth} Exists')
      return Path(pth).read_text(encoding='utf-8')

FOCUS = 'DIALOG-HEAVY,ACTION-HEAVY,EMOTIONAL SUBTEXT,MULTI-CHARACTER,PROP PASSING,EXPLORATION,POWER DYNAMIC,INTIMACY ESCALATION,MISUNDERSTANDING,TIME PRESSURE'.split(',')
GENRES = 'Medieval Fantasy,Cyberpunk,Post-Apocalyptic,Victorian,Sci-Fi Space Station,1920s Noir,Modern Urban,Ancient Mythological,Steampunk,Western'.split(',')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output', type=str, default='story.txt')
    parser.add_argument('-F', '--focus', type=str, default=None, help=f'One of {', '.join(FOCUS)}')
    parser.add_argument('-G', '--genre', type=str, default=None, help=f'One of {', '.join(GENRES)}')
    parser.add_argument('-D', '--gender', type=str, default='mixed')
    parser.add_argument('-N', '--number-characters', type=int, default=2, choices=range(1, 4), metavar='N',
                       help='Number of characters (1-3, default: 2)')
    args = parser.parse_args()
    
    # Do random selection in Python, not the LLM
    genre = args.genre if args.genre else random.choice(GENRES)
    focus = args.focus.upper() if args.focus and args.focus.upper() in FOCUS else random.choice(FOCUS)

    # Focuses that physically require more than 1 character
    REQUIRES_MULTI = {'DIALOG-HEAVY', 'PROP PASSING', 'POWER DYNAMIC', 'INTIMACY ESCALATION', 'MISUNDERSTANDING', 'MULTI-CHARACTER'}

    # Determine final character count with hard guardrails
    if focus in REQUIRES_MULTI and args.number_characters < 2:
        print(f"⚠️  Auto-correcting: '{focus}' requires at least 2 characters.")
        char_count = 3 if focus == 'MULTI-CHARACTER' else 2
    else:
        char_count = args.number_characters
    
    inputs = f"Generate a test seed\nGenre: {genre}\nFocus: {focus}"
    SEED_GENERATOR = seed_generator(args.gender, genre, focus, f'exactly {char_count} characters')
    print(run_prompt(inputs, SEED_GENERATOR, args.output))

if __name__ == '__main__':
    main()
