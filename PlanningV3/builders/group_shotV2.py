import sys
import json
import difflib
from pathlib import Path
sys.path.append('./lib')
from qwen_llm import llm_analyze_media

# ---------------------------------------------------------
# POST-PROCESSING: Insert identity tags after character names
# ---------------------------------------------------------

def attach_identity_tags(scenes, registry):
    # Build name → identity lookup
    identity_map = {}
    for c in registry["characters"]:
        name = c["name"]
        identity = c["identity"]  # MUST be "<gender>, <hair silhouette>, <clothing color>"
        identity_map[name] = identity

    # Helper: rewrite a single text field
    def rewrite_text(text, characters_in_shot):
        if not isinstance(text, str):
            return text

        for char in characters_in_shot:
            name = char["name"]
            identity = identity_map.get(name)
            if not identity:
                continue

            # Replace "Samuel" → "Samuel (male, short-hair silhouette, blue clothing)"
            # Only if not already tagged
            if name in text and f"{name} (" not in text:
                text = text.replace(name, f"{name} ({identity})")

        return text

    # Walk all scenes → shots → characters
    for scene in scenes["scenes"]:
        for scene_id, scene_data in scene.items():
            for shot in scene_data["shots"]:
                chars = shot.get("characters", [])

                # Rewrite shot-level action
                if "action" in shot:
                    shot["action"] = rewrite_text(shot["action"], chars)

                # Rewrite dialog lines
                for d in shot.get("dialog", []):
                    d["line"] = rewrite_text(d["line"], chars)

                # Rewrite per-character pose/action
                for c in chars:
                    c["pose"] = rewrite_text(c.get("pose", ""), chars)
                    c["action"] = rewrite_text(c.get("action", ""), chars)

    return scenes


# ---------------------------------------------------------
# LOAD REGISTRY ZONES (SINGLE SOURCE OF TRUTH)
# ---------------------------------------------------------

def load_registry_zone_descriptions(registry_path: str):
    registry = json.loads(Path(registry_path).read_text())
    zones = registry["locations"][0]["zones"]

    slug_to_description = {}
    for z in zones:
        name = z["zone_name"]
        slug = slugify(name)
        desc = z["description"]
        slug_to_description[slug] = desc

    return slug_to_description


import re

def slugify(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '-', text).strip('-').lower()

def load_registry_zones(registry_path: str):
    registry = json.loads(Path(registry_path).read_text())
    locations = registry.get("locations", [])
    if not locations:
        raise ValueError("registry.json has no locations")

    zones = locations[0].get("zones", [])
    if not zones:
        raise ValueError("registry.json has no zones in first location")

    # canonical list of zone names
    zone_name_list = [z["zone_name"] for z in zones]

    # map zone_name → description
    zone_description_map = {
        z["zone_name"]: z["description"]
        for z in zones
    }

    return zone_name_list, zone_description_map




# ---------------------------------------------------------
# FUZZY MATCHING: beat.zone → registry zone_name (stdlib only)
# ---------------------------------------------------------

def fuzzy_match_zone(human_zone: str, all_zone_names):
    if not human_zone:
        return None

    # First pass: direct fuzzy match
    matches = difflib.get_close_matches(human_zone, all_zone_names, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    # Second pass: case-insensitive exact match
    lower_map = {name.lower(): name for name in all_zone_names}
    hz = human_zone.lower()
    if hz in lower_map:
        return lower_map[hz]

    # Third pass: substring-based heuristic
    for key, original in lower_map.items():
        if key in hz or hz in key:
            return original

    return None


# ---------------------------------------------------------
# SCENE SPLITTING (unchanged)
# ---------------------------------------------------------

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
# MAIN EXECUTION — REGISTRY-DRIVEN, FUZZY-MATCHED ZONES
# ---------------------------------------------------------

if __name__ == '__main__':
    base = sys.argv[1]
    out_path = sys.argv[2]

    # Load registry
    registry = json.loads(Path(f"{base}/registry.json").read_text())

    # Load canonical zones
    all_zone_names, zone_description_map = load_registry_zones(
        f"{base}/registry.json"
    )

    # Build canonical character list
    canonical_characters = [
        {"name": c["name"], "identity": c["identity"]}
        for c in registry["characters"]
    ]

    PHASE_1 = Path('./PlanningV3/prompts/groupshot/phase1.txt').read_text()

    scenes = {"scenes": []}

    for scene in iter_scenes_from_complete_json(f"{base}/complete.json"):

        print("SCENE", scene["scene_id"])

        beats = scene["beats"]
        all_shots = []

        for i, curr in enumerate(beats):
            prev = beats[i-1] if i > 0 else ""
            nextb = beats[i+1] if i < len(beats)-1 else ""

            human_zone = curr.get("zone", "")
            canonical_zone_name = fuzzy_match_zone(human_zone, all_zone_names)

            if not canonical_zone_name:
                print(f"WARNING: Could not match zone '{human_zone}'")
                continue

            phase1_input = {
                "previous_beat": prev,
                "current_beat": curr,
                "next_beat": nextb,
                "environment_zone_description": zone_description_map[canonical_zone_name],
                "characters": canonical_characters
            }

            data = llm_analyze_media(
                media="",
                prompt=json.dumps(phase1_input),
                system=PHASE_1,
                max_tokens=4096
            )['analysis']

            try:
                shots_obj = json.loads(data)
                shots = shots_obj["shots"]
            except Exception:
                print("FAILED TO PARSE PHASE 1 OUTPUT")
                print(data)
                sys.exit(1)

            for s in shots:
                s["shot_id"] = len(all_shots) + 1
                s["environment_zone"] = canonical_zone_name
                all_shots.append(s)

        scenes["scenes"].append({
            scene["scene_id"]: {
                "shots": all_shots
            }
        })

    scenes = attach_identity_tags(scenes, registry)

    with open(out_path, 'w') as wr:
        json.dump(scenes, wr, indent=4)
