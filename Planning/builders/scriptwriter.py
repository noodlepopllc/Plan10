import json, sys
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from pathlib import Path

WORLD = Path('./Planning/prompts/scriptwriter/world.txt').read_text()
BIOGRAPHY = Path('./Planning/prompts/scriptwriter/biography.txt').read_text()
ACTION = Path('./Planning/prompts/scriptwriter/action.txt').read_text()
STORY = Path('./Planning/prompts/scriptwriter/story.txt').read_text()
NARRATOR = Path('./Planning/prompts/scriptwriter/narrator.txt').read_text()
VALIDATOR = Path('./Planning/prompts/scriptwriter/validator.txt').read_text()


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



def build_script(story, outpath):
    expanded = Path(f'{outpath}/story.txt').read_text()
    world = run_prompt(f'{expanded}', WORLD, f'{outpath}/world.txt')
    biography = run_prompt(world, BIOGRAPHY, f'{outpath}/biography.txt')
    narrative = run_prompt(f'Biography: {biography}, Story: {expanded}', NARRATOR, f'{outpath}/narrative.txt')

    action_beats = run_prompt(
        f'World: {biography}, Narrative: {narrative}', 
        ACTION,
        f'{outpath}/action_beats.txt')

    validator = run_prompt(action_beats, VALIDATOR, f'{outpath}/validated.txt')

    complete = run_prompt(
        f'World: {biography}, Action Beats: {validator}, Narrative: {narrative}',
        STORY,
        f'{outpath}/complete.json')
    
if __name__ == '__main__':
    from pathlib import Path
    user_input = Path(sys.argv[1]).read_text()
    outpath = sys.argv[2]
    build_script(user_input, outpath)
