import json, sys
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from pathlib import Path

WORLD = Path('./PlanningV4/prompts/scriptwriter/world.txt').read_text()
BIOGRAPHY = Path('./PlanningV4/prompts/scriptwriter/biography.txt').read_text()
ACTION = Path('./PlanningV4/prompts/scriptwriter/action.txt').read_text()
DIALOG = Path('./PlanningV4/prompts/scriptwriter/dialog.txt').read_text()
STORY = Path('./PlanningV4/prompts/scriptwriter/story.txt').read_text()
NARRATOR = Path('./PlanningV4/prompts/scriptwriter/narratorV2.txt').read_text()


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
    world = run_prompt(story, WORLD, f'{outpath}/world.txt')
    biography = run_prompt(world, BIOGRAPHY, f'{outpath}/biography.txt')
    narrative = run_prompt(f'Biography: {biography}, Story: {story}', NARRATOR, f'{outpath}/narrative.txt')

    action_beats = run_prompt(
        f'World: {biography}, Narrative: {narrative}', 
        ACTION,
        f'{outpath}/action_beats.txt')

    dialog_beats = run_prompt(
        f'World: {biography}, Story Beats: {action_beats}, Narrative: {narrative}', 
        DIALOG,
        f'{outpath}/dialog_beats.txt')

    complete = run_prompt(
        f'World: {biography}, Action Beats: {action_beats}, Dialog Beats: {dialog_beats}, Narrative: {narrative}',
        STORY,
        f'{outpath}/complete.json')
    
if __name__ == '__main__':
    from pathlib import Path
    user_input = Path(sys.argv[1]).read_text()
    outpath = sys.argv[2]
    build_script(user_input, outpath)
