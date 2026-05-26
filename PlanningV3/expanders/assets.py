import json
import re
from collections import defaultdict


def slugify(text):
    return re.sub(r'[^a-zA-Z0-9]+', '-', text).strip('-')


def zone_slug_from_shot(shot, scene_id):
    zone_text = shot.get("environment_zone") or shot.get("location") or "Zone"
    first_word = zone_text.split()[0]
    return f"{scene_id}_{first_word.capitalize()}" if scene_id else f"{first_word.capitalize()}"


def find_shot(shots, shot_id):
    for s in shots:
        if s["shot_id"] == shot_id:
            return s
    raise KeyError(f"Shot {shot_id} not found")


def extract_emotion(appearance_text):
    if not appearance_text:
        return "Neutral"

    text = appearance_text.lower()

    emotion_keywords = {
        "analytical": "Analytical",
        "rigid": "Rigid",
        "stoic": "Stoic",
        "commanding": "Commanding",
        "expressive": "Expressive",
        "overwhelmed": "Overwhelmed",
        "questioning": "Questioning",
        "vulnerable": "Vulnerable",
        "resigned": "Resigned"
    }

    for key, val in emotion_keywords.items():
        if key in text:
            return val

    return "Neutral"


def build_i2v_videos(scene_id, assets, shots):
    video_nodes = []
    shot_map = {str(s["shot_id"]): s for s in shots}

    for asset in assets:
        # Only generate I2V for shot composites
        if not asset["alias"].startswith(f"{scene_id}_SHOT_"):
            continue

        match = re.search(r'SHOT_(\d+)$', asset["alias"])
        if not match:
            continue

        shot_id = match.group(1)
        shot = shot_map.get(shot_id)
        if not shot:
            continue

        # Micro‑motion summary
        micro_motion = ", ".join([
            c.get("appearance_in_shot", "").strip().rstrip(".")
            for c in shot["characters"]
            if c.get("appearance_in_shot")
        ]) or "subtle environmental motion"

        # Build instruction
        instruction = (
            f"create an image_to_video using {asset['alias']} asset, "
            f"action: {micro_motion}"
        )

        video_nodes.append({
            "alias": f"{scene_id}_VID_{shot_id}",
            "alias_used": [asset["alias"]],
            "instruction": instruction
        })

    return video_nodes


def build_dependency_graph(registry, scene_id, shots):
    graph = {
        "scene_id": scene_id,
        "identity": [],
        "background": None,
        "zone_backdrops": [],     # NEW: zone × layout × camera_side
        "base_composites": [],    # NEW: character-in-zone
        "shot_composites": [],
        "dialog": []
    }

    # ---------------------------------------------------------
    # IDENTITY
    # ---------------------------------------------------------
    for c in registry["characters"]:
        name = c["name"].upper()
        graph["identity"].append({
            "alias": f"{name}_Sheet",
            "dependencies": [],
            "character": name
        })
        graph["identity"].append({
            "alias": f"{name}_Voice",
            "dependencies": [],
            "character": name
        })

    # ---------------------------------------------------------
    # BACKGROUND (scene-level)
    # ---------------------------------------------------------
    def resolve_zone_description(registry, zone_name):
        loc = registry["locations"][0]
        for z in loc["zones"]:
            if zone_name in z["zone_name"]:
                return z["description"]
        return registry["world"]["description"]  # fallback interior description

    primary_zone = shots[0]["environment_zone"]
    env_prompt = resolve_zone_description(registry, primary_zone)

    bg_alias = f"{scene_id}_BG"
    graph["background"] = {
        "alias": bg_alias,
        "dependencies": [],
        "prompt": env_prompt
    }


    # ---------------------------------------------------------
    # ZONE BACKDROPS (zone × layout × camera_side)
    # ---------------------------------------------------------
    backdrop_cache = {}   # (zone, layout, camera_side) -> alias

    for shot in shots:
        zone = shot["environment_zone"]
        layout = shot.get("layout", "solo")
        camera_side = shot.get("camera_side", "center")

        key = (zone, layout, camera_side)

        if key in backdrop_cache:
            zb_alias = backdrop_cache[key]
        else:
            zb_alias = f"{scene_id}_ZB_{len(backdrop_cache) + 1}"
            backdrop_cache[key] = zb_alias

            cv = shot.get("camera_view", {})
            prompt = (
                f"Environment zone: {zone}. "
                f"Layout: {layout}. "
                f"Camera side: {camera_side}. "
                f"Shot-level camera view: "
                f"angle={cv.get('angle')}, "
                f"height={cv.get('height')}, "
                f"distance={cv.get('distance')}, "
                f"framing={cv.get('framing')}, "
                f"facing={cv.get('facing')}."
            )

            graph["zone_backdrops"].append({
                "alias": zb_alias,
                "dependencies": [bg_alias],
                "zone": zone,
                "layout": layout,
                "camera_side": camera_side,
                "prompt": prompt
            })

        shot["_zone_backdrop_alias"] = zb_alias  # stash for later stages

    # ---------------------------------------------------------
    # BASE COMPOSITES (character-in-zone)
    # ---------------------------------------------------------
    for shot in shots:
        zb_alias = shot["_zone_backdrop_alias"]

        for char in shot["characters"]:
            name = char["name"].upper()
            base_alias = f"{scene_id}_BASE_{shot['shot_id']}_{name}"

            graph["base_composites"].append({
                "alias": base_alias,
                "dependencies": [
                    f"{name}_Sheet",
                    zb_alias
                ],
                "character": name,
                "shot_id": shot["shot_id"],
                "zone": shot["environment_zone"],
                "layout": shot.get("layout", "solo"),
                "camera_side": shot.get("camera_side", "center"),
                "character_camera_view": char.get("camera_view", {}),
            })

    # ---------------------------------------------------------
    # SHOT COMPOSITES (camera transforms over base composites)
    # ---------------------------------------------------------
    IMAGE_SHOT_TYPES = [
        "closeup", "medium", "wide", "two_shot", "ots", "profile",
        "establishing", "tracking", "insert", "cutaway", "transition"
    ]

    for shot in shots:
        if shot["type"] in IMAGE_SHOT_TYPES:
            sc_alias = f"{scene_id}_SHOT_{shot['shot_id']}"

            deps = [
                f"{scene_id}_BASE_{shot['shot_id']}_{c['name'].upper()}"
                for c in shot["characters"]
            ]

            graph["shot_composites"].append({
                "alias": sc_alias,
                "dependencies": deps,
                "description": shot["description"],
                "characters": [c["name"] for c in shot["characters"]],
                "camera_view": shot.get("camera_view", {}),
                "environment_zone": shot["environment_zone"],
                "layout": shot.get("layout", "solo"),
                "camera_side": shot.get("camera_side", "center"),
                "camera_focus": shot.get("camera_focus", ""),
                "shot_id": shot["shot_id"]
            })

    # ---------------------------------------------------------
    # DIALOG (depends on shot composite)
    # ---------------------------------------------------------
    for shot in shots:
        dialog_lines = shot.get("dialog", [])
        if not dialog_lines:
            continue

        for i, line in enumerate(dialog_lines):
            speaker = line["speaker"].upper()
            d_alias = f"{scene_id}_D_{shot['shot_id']}_{i}"
            sc_alias = f"{scene_id}_SHOT_{shot['shot_id']}"

            graph["dialog"].append({
                "alias": d_alias,
                "dependencies": [f"{speaker}_Voice", sc_alias],
                "speaker": speaker,
                "shot_id": shot["shot_id"],
                "line_index": i
            })

    return graph



def generate_assets(registry, shots, graph):
    assets = []
    registry_map = {c["name"].upper(): c for c in registry["characters"]}

    # ---------------------------------------------------------
    # IDENTITY → create_character_sheet / design_voice
    # ---------------------------------------------------------
    for item in graph["identity"]:
        name = item["character"]
        char = registry_map[name]

        if item["alias"].endswith("_Sheet"):
            prompt = char["appearance_prompt"]
            instruction = (
                f"create_character_sheet using prompt: {prompt}"
            )
        else:
            voice_desc = char["voice"]
            instruction = (
                f"design_voice using voice: {voice_desc}"
            )

        assets.append({
            "alias": item["alias"],
            "alias_used": item["dependencies"],
            "instruction": instruction
        })

    # ---------------------------------------------------------
    # BACKGROUND → create_background
    # ---------------------------------------------------------
    bg = graph["background"]
    bg_prompt = bg["prompt"]

    bg_instruction = (
        f"create_background using prompt: {bg_prompt}"
    )

    assets.append({
        "alias": bg["alias"],
        "alias_used": [],
        "instruction": bg_instruction
    })

    # ---------------------------------------------------------
    # ZONE BACKDROPS → generate_backdrop
    # ---------------------------------------------------------
    for zb in graph["zone_backdrops"]:
        bg_alias = zb["dependencies"][0]

        instruction = (
            f"generate_backdrop using media: {bg_alias} asset, "
            f"zone: {zb['zone']}"
        )

        assets.append({
            "alias": zb["alias"],
            "alias_used": zb["dependencies"],
            "instruction": instruction
        })

    # ---------------------------------------------------------
    # BASE COMPOSITES → composite_scene
    # ---------------------------------------------------------
    for base in graph["base_composites"]:
        name = base["character"]
        sheet_alias = f"{name}_Sheet"
        zb_alias = base["dependencies"][1]

        instruction = (
            f"create a composite by combining {sheet_alias} asset with {zb_alias} asset, "
            f"neutral placement in zone {base['zone']}"
        )

        assets.append({
            "alias": base["alias"],
            "alias_used": base["dependencies"],
            "instruction": instruction
        })

    # ---------------------------------------------------------
    # SHOT COMPOSITES → apply_gimbal_shot
    # ---------------------------------------------------------
    for sc in graph["shot_composites"]:
        shot = find_shot(shots, sc["shot_id"])
        shot_cv = sc.get("camera_view", {}) or {}

        if not sc["dependencies"]:
            continue

        source_media = sc["dependencies"][0]

        angle = shot_cv.get("angle", "front")
        height = shot_cv.get("height", "eye")
        distance = shot_cv.get("distance", "medium")

        instruction = (
            f"apply gimbal shot to {source_media} asset, "
            f"angle: {angle}, height: {height}, distance: {distance}"
        )

        assets.append({
            "alias": sc["alias"],
            "alias_used": sc["dependencies"],
            "instruction": instruction
        })

    # ---------------------------------------------------------
    # DIALOG → dialog_to_video
    # ---------------------------------------------------------
    for d in graph["dialog"]:
        shot = find_shot(shots, d["shot_id"])
        line = shot["dialog"][d["line_index"]]["line"]
        speaker = d["speaker"]

        sc_alias = f"{graph['scene_id']}_SHOT_{d['shot_id']}"
        media_alias = sc_alias
        audio_alias = f"{speaker}_Voice.wav"

        instruction = (
            f"create a dialog_to_video using {audio_alias} asset with {media_alias} asset, "
            f"text: \"{line}\""
        )

        assets.append({
            "alias": d["alias"],
            "alias_used": d["dependencies"],
            "instruction": instruction
        })

    return assets



if __name__ == '__main__':
    import sys
    from pathlib import Path

    basepath = sys.argv[1]
    registry = json.loads(Path(f'{basepath}/registry.json').read_text())
    scene_id = sys.argv[2]

    scenes_data = json.loads(Path(f'{basepath}/shots.json').read_text())["scenes"]

    shots = None
    for s in scenes_data:
        if scene_id in s:
            shots = s[scene_id]["shots"]
            break
        if f"Scene{scene_id}" in s:
            shots = s[f"Scene{scene_id}"]["shots"]
            break

    if shots is None:
        raise KeyError(f"No scene found for {scene_id}")

    graph = build_dependency_graph(registry, scene_id, shots)
    assets = generate_assets(registry, shots, graph)
    i2v = build_i2v_videos(scene_id, assets, shots)
    assets += i2v

    with open(f'{basepath}/assets{scene_id}.json', 'w') as output:
        json.dump(assets, output, indent=4)
