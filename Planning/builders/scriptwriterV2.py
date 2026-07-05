import json, sys
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from pathlib import Path

prompt_path = './Planning/prompts'
WORLD = Path(f'{prompt_path}/scriptwriter/worldV2.txt').read_text()
BIOGRAPHY = Path(f'{prompt_path}/scriptwriter/biography.txt').read_text()
NARRATOR = Path(f'{prompt_path}/scriptwriter/narrator.txt').read_text()
ENHANCER = Path(f'{prompt_path}/scriptwriter/enhancer.txt').read_text()  # NEW

REQUIRED_FIELDS = ['actor', 'speaker', 'action', 'dialog', 'location', 'zone', 'backdrop', 'posture', 'facial']
COLD_OPEN_MARKER = '******* COLD OPEN END ****'

def run_prompt(prompt, system, pth):
    if not Path(pth).exists():
      result = llm_analyze_media(
          media="", 
          prompt=prompt,
          system=system,
          max_tokens=8192,
          temperature=0.2)['analysis']
      with open(pth, 'w') as out_f:
        out_f.write(result)
      print(f'Wrote {pth}')
      return result
    else:
      print(f'{pth} Exists')
      return Path(pth).read_text()

def split_story_at_cold_open(prose):
    if COLD_OPEN_MARKER in prose:
        parts = prose.split(COLD_OPEN_MARKER, 1)
        cold_open = parts[0].strip()
        story = parts[1].strip() if len(parts) > 1 else ''
        return cold_open, story
    else:
        print(f'Warning: Cold open marker not found, processing entire story')
        return '', prose

def split_into_paragraphs(prose):
    paragraphs = prose.split('\n\n')
    if len(paragraphs) == 1:
        paragraphs = prose.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def enhance_paragraph(paragraph, cold_open, full_story, current_state):
    """
    Rewrite a paragraph with full narrative context so it's richer
    and more coherent before JSON extraction.
    """
    prompt = f"""You are enhancing a single paragraph of a story. You have the full context.

COLD OPEN (establishes tone/setting):
{cold_open}

FULL STORY (for overall arc context):
{full_story}

CURRENT STATE (where we are in the narrative):
{json.dumps(current_state)}

PARAGRAPH TO ENHANCE:
{paragraph}

TASK: Rewrite this paragraph to be more vivid, coherent, and cinematically detailed. 
Preserve all facts, dialogue, and character actions exactly. 
Do NOT add new plot points. Just improve prose quality, sensory detail, and narrative flow.
Output ONLY the enhanced paragraph text, nothing else."""
    
    result = llm_analyze_media(
        media="",
        prompt=prompt,
        system=ENHANCER,
        max_tokens=2048,
        temperature=0.3
    )['analysis']
    
    return result.strip()

# ... keep fix_truncated_json, validate_beat, parse_jsonl, extract_initial_backdrop, extract_initial_state unchanged ...

def build_script(story, outpath):
    expanded = Path(f'{outpath}/story.txt').read_text()
    world = run_prompt(f'{expanded}', WORLD, f'{outpath}/world.txt')
    biography = run_prompt(world, BIOGRAPHY, f'{outpath}/registry.json')
    
    narrative_path = f'{outpath}/narrative.json'
    if not Path(narrative_path).exists():
        cold_open, story_text = split_story_at_cold_open(expanded)
        
        if cold_open:
            print(f'Found cold open ({len(cold_open)} chars), will use as context')
        else:
            print('No cold open marker found')
        
        paragraphs = split_into_paragraphs(story_text)
        print(f'Split story into {len(paragraphs)} paragraphs')
        
        current_state = extract_initial_state(biography)
        all_beats = []
        
        for idx, paragraph in enumerate(paragraphs):
            print(f'Processing paragraph {idx+1}/{len(paragraphs)}')
            
            # NEW: Enhance paragraph with full context first
            enhanced = enhance_paragraph(paragraph, cold_open, story_text, current_state)
            print(f'Enhanced paragraph {idx+1} ({len(enhanced)} chars)')
            
            paragraph_prompt = f"""Biography: {biography}

Current State: {json.dumps(current_state)}

Paragraph: {enhanced}"""
            
            result = llm_analyze_media(
                media="",
                prompt=paragraph_prompt,
                system=NARRATOR,
                max_tokens=4096,
                temperature=0.2
            )['analysis']
            
            beats = parse_jsonl(result, current_state)
            all_beats.extend(beats)
            
            if beats:
                last_beat = beats[-1]
                current_state = {
                    "posture": last_beat.get("posture", current_state["posture"]),
                    "facial": last_beat.get("facial", current_state["facial"]),
                    "location": last_beat.get("location", current_state["location"]),
                    "zone": last_beat.get("zone", current_state["zone"]),
                    "backdrop": last_beat.get("backdrop", current_state["backdrop"]),
                    "last_actor": last_beat.get("actor", current_state["last_actor"])
                }
        
        with open(narrative_path, 'w') as out_f:
            for beat in all_beats:
                out_f.write(json.dumps(beat) + '\n')
        
        print(f'Wrote {narrative_path} with {len(all_beats)} beats')
    else:
        print(f'{narrative_path} Exists')

if __name__ == '__main__':
    user_input = Path(sys.argv[1]).read_text()
    outpath = sys.argv[2]
    build_script(user_input, outpath)