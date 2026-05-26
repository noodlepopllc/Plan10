import re, sys
from pathlib import Path
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from json import loads, dump
import json
from pathlib import Path

import json
import difflib
from pathlib import Path

def normalize_zone_name(name: str):
    # Remove leading numbering like "1. " or "2. "
    return re.sub(r"^\d+\.\s*", "", name).strip()


def patch_registry_zones(registry_path: str, canonical_zones: dict):
    registry = json.loads(Path(registry_path).read_text())

    canonical_names = list(canonical_zones.values())

    for loc in registry.get("locations", []):
        for zone in loc.get("zones", []):
            name = zone.get("zone_name") or zone.get("name") or ""
            if not name:
                continue

            # fuzzy match against canonical human-readable names
            match = difflib.get_close_matches(name, canonical_names, n=1, cutoff=0.7)
            if not match:
                print(f"[registry patch] no match for zone '{name}'")
                continue

            canonical_name = match[0]

            # find slug for that canonical name
            slug = None
            for s, human in canonical_zones.items():
                if human == canonical_name:
                    slug = s
                    break

            if not slug:
                print(f"[registry patch] no slug for matched zone '{canonical_name}'")
                continue

            zone["zone_name"] = canonical_name
            zone["slug"] = slug

    Path(registry_path).write_text(json.dumps(registry, indent=4))

def generate_canonical_zone_dictionary(pass_a_locations_text):
    zones = {}

    # Match numbered markdown headings like "**2. Common Tables**"
    zone_pattern = re.compile(r"\*\*\s*\d+\.\s*(.*?)\s*\*\*")
    matches = zone_pattern.findall(pass_a_locations_text)

    for zone_name in matches:
        zone_name = normalize_zone_name(zone_name)

        slug = re.sub(r"[^a-z0-9]+", "_", zone_name.lower()).strip("_")
        zones[slug] = zone_name

    return zones




def extract_zones_from_pass_a(pass_a_text: str):
    """
    Extract canonical zone names from PASS A and return:
    {
        "kitchen_counter_area": "Kitchen Counter Area",
        ...
    }
    """
    zones = generate_canonical_zone_dictionary(pass_a_text)
    return zones


def pre_match_zone_for_beat(beat_text: str, canonical_zones: dict):
    """
    Given a beat's action/dialog text, match it to the closest canonical zone.
    This is a simple keyword-based matcher.
    """

    text = beat_text.lower()

    # Try exact keyword matches first
    for slug, name in canonical_zones.items():
        key = name.lower().split(" area")[0].split(" zone")[0]
        key = key.replace(" ", "_")
        if key in text.replace(" ", "_"):
            return slug

    # Try partial matches
    for slug, name in canonical_zones.items():
        words = name.lower().split()
        if any(w in text for w in words):
            return slug

    # Fallback: return None (Phase 1 will error if missing)
    return None


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

def split_scenes_by_zone(beats):
    scenes = []
    current_scene = []
    current_zone = beats[0]["zone"]

    for beat in beats:
        if beat["zone"] != current_zone:
            scenes.append({
                "zone": current_zone,
                "beats": current_scene
            })
            current_scene = []
            current_zone = beat["zone"]

        current_scene.append(beat)

    scenes.append({
        "zone": current_zone,
        "beats": current_scene
    })

    return scenes


def iter_scenes_from_complete_json(path: str):
    beats = json.loads(Path(path).read_text())
    scenes = split_scenes_by_zone(beats)

    for i, scene in enumerate(scenes, start=1):
        yield {
            "scene_id": i,
            "zone": scene["zone"],
            "beats": scene["beats"]
        }




# ---------------------------------------------------------
# MAIN EXECUTION — PATCHED FOR 3-BEAT TEMPORAL CONTINUITY
# ---------------------------------------------------------

if __name__ == '__main__':
    base = sys.argv[1]
    out_path = sys.argv[2]

    biography = Path(f'{base}/biography.txt').read_text()

    # NEW: extract canonical zones from PASS A
    canonical_zones = extract_zones_from_pass_a(biography)
    print(canonical_zones)

    scenes = {"scenes": []}

    for scene in iter_scenes_from_complete_json(f"{base}/complete.json"):

        print("SCENE", scene["scene_id"])

        beats = scene["beats"]
        all_shots = []

        for i, curr in enumerate(beats):
            prev = beats[i-1] if i > 0 else ""
            nextb = beats[i+1] if i < len(beats)-1 else ""

            # NEW: pre-match zone for this beat
            actions = curr.get("actions", [])
            dialog = curr.get("dialog", [])

            # Flatten actions into a single string
            action_text = " ".join(actions)

            # Flatten dialog lines into a single string
            dialog_text = " ".join(d["line"] for d in dialog)

            beat_text = (action_text + " " + dialog_text).strip()

            zone_slug = pre_match_zone_for_beat(beat_text, canonical_zones)

            if not zone_slug:
                print("WARNING: Could not match zone for beat:", beat_text)

            phase1_input = {
                "previous_beat": prev,
                "current_beat": curr,
                "next_beat": nextb,
                "biography": biography,

                # NEW: inject canonical zone
                "environment_zone": zone_slug
            }

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

            for s in shots:
                s["shot_id"] = len(all_shots) + 1
                all_shots.append(s)

        scenes["scenes"].append({
            scene["scene_id"]: {
                "shots": all_shots
            }
        })

    with open(out_path, 'w') as wr:
        dump(scenes, wr, indent=4)

    patch_registry_zones(f'{base}/registry.json', canonical_zones)
