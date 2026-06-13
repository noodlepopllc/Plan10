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

DETAIL EXPANSION RULE (MANDATORY)

All narration MUST include rich, concrete physical detail.
You MUST describe posture, movement, gestures, facial cues, and spatial relationships.
You MUST describe the environment, anchored objects, and how characters interact with them.
You MUST expand each beat into a fully realized cinematic moment.
You MUST NOT leave beats minimal or vague.
You MUST NOT omit physical blocking or emotional cues.
"""

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--story', type=str, default='')
    parser.add_argument('-O', '--output', type=str, default='story.txt')
    args = parser.parse_args()
    story = Path(args.story).read_text() if args.story else SEED
    print(run_prompt(story, SYSTEM, args.output))
