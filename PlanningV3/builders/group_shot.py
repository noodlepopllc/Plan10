import re, sys
from pathlib import Path
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from json import loads, dump
import json
from pathlib import Path

PHASE_1 = Path('./PlanningV3/prompts/groupshot/phase1.txt').read_text()
PHASE_2 = Path('./PlanningV3/prompts/groupshot/phase2.txt').read_text()

SCENE_RE = re.compile(
    r"<SCENE id=\"(\d+)\">(.+?)</SCENE>",
    re.DOTALL
)

SLUGLINE_RE = re.compile(
    r"<SLUGLINE>(.+?)</SLUGLINE>"
)

# <BEAT type="action">...</BEAT>
# <BEAT type="dialog" speaker="ELARA">...</BEAT>
BEAT_RE = re.compile(
    r"<BEAT\s+type=\"(action|dialog)\"(?:\s+speaker=\"([A-Z0-9_]+)\")?>(.+?)</BEAT>",
    re.DOTALL
)

# ---------------------------------------------------------
# ITERATE SCENES — NOW RETURNS BEATS AS A LIST
# ---------------------------------------------------------

def iter_scenes_from_xml(script_text: str):
    for m in SCENE_RE.finditer(script_text):
        scene_id = int(m.group(1))
        scene_body = m.group(2)

        slug = SLUGLINE_RE.search(scene_body)
        slugline = slug.group(1).strip() if slug else ""

        beats = []

        for beat_type, speaker, content in BEAT_RE.findall(scene_body):
            content = content.strip()
            if not content:
                continue

            if beat_type == "action":
                beats.append(content)
            else:
                if speaker:
                    beats.append(f"{speaker}:\n{content}")
                else:
                    beats.append(content)

        yield {
            "scene_id": scene_id,
            "scene_heading": slugline,
            "beats": beats
        }

# ---------------------------------------------------------
# MAIN EXECUTION — PATCHED FOR 3-BEAT TEMPORAL CONTINUITY
# ---------------------------------------------------------

if __name__ == '__main__':
    base = sys.argv[1]
    out_path = sys.argv[2]

    text = Path(f'{base}/screenplay.txt').read_text()
    biography = Path(f'{base}/registry.json').read_text()
    biography_json = loads(biography)

    scenes = {"scenes": []}

    for scene in iter_scenes_from_xml(text):
        print("SCENE", scene["scene_id"])

        beats = scene["beats"]
        all_shots = []

        # ---------------------------------------------
        # 3-BEAT WINDOW LOOP
        # ---------------------------------------------
        for i, curr in enumerate(beats):
            prev = beats[i-1] if i > 0 else ""
            nextb = beats[i+1] if i < len(beats)-1 else ""

            phase1_input = {
                "previous_beat": prev,
                "current_beat": curr,
                "next_beat": nextb,
                "biography": biography_json
            }

            # -----------------------------------------
            # CALL PHASE 1 FOR THIS BEAT
            # -----------------------------------------
            data = llm_analyze_media(
                media="",
                prompt=json.dumps(phase1_input),
                system=PHASE_1,
                max_tokens=4096
            )['analysis']

            try:
                shots_obj = loads(data)
                shots = shots_obj["shots"]
            except Exception as e:
                print("FAILED TO PARSE PHASE 1 OUTPUT")
                print(data)
                sys.exit()

            '''
            # -----------------------------------------
            # CALL PHASE 2 FOR EACH SHOT
            # -----------------------------------------
            for shot in shots:
                data_2 = llm_analyze_media(
                    media="",
                    prompt=json.dumps(shot),
                    system=PHASE_2,
                    max_tokens=4096
                )['analysis']

                try:
                    detailed = loads(data_2)
                    shot.update(detailed)
                except:
                    print("Broken PHASE 2 output:", data_2)
                    sys.exit()
            '''

            for s in shots:
                s["shot_id"] = len(all_shots) + 1
                all_shots.append(s)

        # ---------------------------------------------
        # STORE SCENE
        # ---------------------------------------------
        scenes["scenes"].append({
            scene["scene_id"]: {
                "shots": all_shots
            }
        })

    # -------------------------------------------------
    # WRITE OUTPUT
    # -------------------------------------------------
    with open(out_path, 'w') as wr:
        dump(scenes, wr, indent=4)
