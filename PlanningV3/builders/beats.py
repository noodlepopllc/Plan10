import re

SCENE_MARKER_RE = re.compile(
    r'^\*\*SCENE\s+(?:\d+|START)(?:[^\*]*)\*\*$',
    re.MULTILINE
)


def iter_scenes_by_marker(script_text: str):
    """
    Yields scenes as {scene_id, scene_heading, scene_text} based on **SCENE N** markers.
    Everything before the first **SCENE** is ignored.
    """
    matches = list(SCENE_MARKER_RE.finditer(script_text))
    if not matches:
        return  # no scenes found

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(script_text)

        block = script_text[start:end].strip()
        lines = block.split("\n", 1)

        scene_heading = lines[0].strip()              # e.g. **SCENE 1**
        scene_text = lines[1].strip() if len(lines) > 1 else ""

        # extract numeric id from heading
        m_id = re.search(r'\bSCENE\s+(\d+)', scene_heading)
        scene_id = int(m_id.group(1)) if m_id else i + 1

        yield {
            "scene_id": scene_id,
            "scene_heading": scene_heading,
            "scene_text": scene_text
        }



if __name__ == '__main__':
    import sys
    sys.path.append('./lib')
    from pathlib import Path
    from qwen_llm import llm_analyze_media
    from pprint import pprint
    from json import loads, dump

    text = Path(sys.argv[1]).read_text()
    scenes = {"scenes":[]}
    for i, scene in enumerate(iter_scenes_by_marker(text)):
        print("SCENE ",i+1)
        system_prompt = Path('./PlanningV3/prompts/beats.txt').read_text().strip()
        for x in range(5):
            data = "".join([x.strip() for x in llm_analyze_media('',scene['scene_text'],system_prompt,8192)['analysis']])
            try:
                a = loads(data)
                scenes["scenes"].append({(i+1): a})
                break
            except:
                print(scene)
                print(data)
    with open(sys.argv[2],'w') as wr:
        dump(scenes, wr, indent=4)
