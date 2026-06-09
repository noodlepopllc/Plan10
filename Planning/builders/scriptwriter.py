import json, sys
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from pathlib import Path
prompt_path = './Planning/prompts'
WORLD = Path(f'{prompt_path}/scriptwriter/world.txt').read_text()
BIOGRAPHY = Path(f'{prompt_path}/scriptwriter/biography.txt').read_text()
NARRATOR = Path(f'{prompt_path}/scriptwriter/narrator.txt').read_text()

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
    narrative = run_prompt(f'Biography: {biography}, Story: {expanded}', NARRATOR, f'{outpath}/narrative.json')
    
if __name__ == '__main__':
    from pathlib import Path
    user_input = Path(sys.argv[1]).read_text()
    outpath = sys.argv[2]
    build_script(user_input, outpath)
