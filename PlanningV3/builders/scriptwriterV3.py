import json, sys
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from pathlib import Path

WORLD = Path('./PlanningV3/prompts/scriptwriter/world.txt').read_text()
BIOGRAPHY = Path('./PlanningV3/prompts/scriptwriter/biography.txt').read_text()
ACTION = Path('./PlanningV3/prompts/scriptwriter/action.txt').read_text()
DIALOG = Path('./PlanningV3/prompts/scriptwriter/dialog.txt').read_text()
STORY = Path('./PlanningV3/prompts/scriptwriter/story.txt').read_text()
TIMELINE = Path('./PlanningV3/prompts/scriptwriter/timeline.txt').read_text()
SCRIPT = Path('./PlanningV3/prompts/scriptwriter/script.txt').read_text()
FORMAT = Path('./PlanningV3/prompts/scriptwriter/format.txt').read_text()
PLANNER = Path('./PlanningV3/prompts/scriptwriter/planner.txt').read_text()


def run_prompt(prompt, system):
    return llm_analyze_media(
        media="", 
        prompt=prompt,
        system=system,
        max_tokens=8192)['analysis']

def build_script(story, outpath):

    world = run_prompt(story, WORLD)
    with open(f'{outpath}/world.txt', 'w') as world_f:
      world_f.write(world)
    print('Wrote World')

    biography = run_prompt(world, BIOGRAPHY)
    with open(f'{outpath}/biography.txt', 'w') as biography_f:
      biography_f.write(biography)
    print('Wrote Biography')

    action_beats = run_prompt(
        f'World: {biography}, Narrative: {story}', 
        ACTION)
    with open(f'{outpath}/action_beats.txt', 'w') as action_beats_f:
      action_beats_f.write(action_beats)
    print('Wrote Action Beats')

    dialog_beats = run_prompt(
        f'World: {biography}, Story Beats: {action_beats}, Narrative: {story}', 
        DIALOG)
    with open(f'{outpath}/dialog_beats.txt', 'w') as dialog_beats_f:
      dialog_beats_f.write(dialog_beats)
    print('Wrote Dialog Beats')

    complete = run_prompt(
        f'World: {biography}, Action Beats: {action_beats}, Dialog Beats: {dialog_beats}, Narrative: {story}',
        STORY)
    with open(f'{outpath}/complete.txt', 'w') as complete_f:
      complete_f.write(complete)
    print('Wrote Story COMPLETE')

    draft = run_prompt(complete, SCRIPT)
    with open(f'{outpath}/draft.txt', 'w') as draft_f:
        draft_f.write(draft)
    print('Wrote Draft')

    final = run_prompt(draft, FORMAT)
    with open(f'{outpath}/screenplay.txt', 'w') as final_f:
        final_f.write(final)
    print('Wrote Formatted Screenplay')

    '''
    timeline = run_prompt(complete, TIMELINE)
    with open(f'{outpath}/timeline.txt', 'w') as timeline_f:
        timeline_f.write(timeline)
    print('Wrote Timeline')

    shot_list = run_prompt(
        f"Biography: {biography}, Script: {final}, Timeline: {timeline}",
        PLANNER)
    with open(f'{outpath}/shot_list.txt', 'w') as shot_list_f:
        shot_list_f.write(shot_list)
    print('Wrote Formatted Screenplay')
    '''

    
if __name__ == '__main__':
    from pathlib import Path
    user_input = Path(sys.argv[1]).read_text()
    outpath = sys.argv[2]
    build_script(user_input, outpath)






    