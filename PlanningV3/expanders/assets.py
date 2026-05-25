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
        if not asset["alias"].startswith(f"{scene_id}_SHOT_"):
            continue

        match = re.search(r'SHOT_(\d+)$', asset["alias"])
        if not match:
            continue

        shot_id = match.group(1)
        shot = shot_map.get(shot_id)
        if not shot:
            continue

        micro_motion = ", ".join([
            c.get("appearance_in_shot", "").strip().rstrip(".")
            for c in shot["characters"]
            if c.get("appearance_in_shot")
        ]) or "subtle environmental motion (heat shimmer, light drift, fabric flutter)"

        # Per-character camera geometry
        char_views = []
        for c in shot["characters"]:
            cv = c.get("camera_view", {})
            char_views.append(
                f"{c['name']}: angle={cv.get('angle')}, "
                f"height={cv.get('height')}, "
                f"distance={cv.get('distance')}, "
                f"framing={cv.get('framing')}, "
                f"facing={cv.get('facing')}"
            )
        char_view_summary = "; ".join(char_views)

        shot_cv = shot.get("camera_view", {})

        instruction = (
            f"Animate this composite image using I2V. "
            f"Scene description: {shot['description']}, "
            f"Action: {shot['action']}, "
            f"Shot framing: angle={shot_cv.get('angle')}, height={shot_cv.get('height')}, "
            f"distance={shot_cv.get('distance')}, framing={shot_cv.get('framing')}, "
            f"facing={shot_cv.get('facing')}. "
            f"Per-character camera geometry: {char_view_summary}. "
            f"Camera focus: {shot['camera_focus']}. "
            f"Micro-motion: {micro_motion}. "
            "Facial expressions only for showing emotion."
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
        "shot_backdrops": [],
        "closeups": [],
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
    # BACKGROUND
    # ---------------------------------------------------------
    env_prompt = shots[0]["environment_zone"]
    bg_alias = f"{scene_id}_BG"

    graph["background"] = {
        "alias": bg_alias,
        "dependencies": [],
        "prompt": env_prompt
    }

    # ---------------------------------------------------------
    # SHOT BACKDROPS
    # ---------------------------------------------------------
    for shot in shots:
        cv = shot.get("camera_view", {})
        zone_anchor = cv.get("zone_anchor", shot.get("environment_zone", shot.get("location", "Unknown zone")))
        sb_alias = f"{scene_id}_ZB_SHOT_{shot['shot_id']}"

        prompt = (
            f"Environment zone: {shot['environment_zone']}. "
            f"Zone anchor: {zone_anchor}. "
            f"Camera view: angle={cv.get('angle', 'front')}, "
            f"height={cv.get('height', 'eye-level')}, "
            f"distance={cv.get('distance', 'medium')}, "
            f"framing={cv.get('framing', shot.get('type', 'medium'))}, "
            f"facing={cv.get('facing', 'toward-character')}."
        )

        graph["shot_backdrops"].append({
            "alias": sb_alias,
            "dependencies": [bg_alias],
            "shot_id": shot["shot_id"],
            "prompt": prompt
        })

    # ---------------------------------------------------------
    # CLOSEUPS
    # ---------------------------------------------------------
    for shot in shots:
        for char in shot["characters"]:
            name = char["name"].upper()
            emotion = extract_emotion(char.get("appearance_in_shot", ""))
            zone = shot["environment_zone"]

            cv = char.get("camera_view") or shot.get("camera_view", {})

            cu_alias = f"{scene_id}_CU_{shot['shot_id']}_{name}"

            graph["closeups"].append({
                "alias": cu_alias,
                "dependencies": [f"{name}_Sheet", bg_alias],
                "character": name,
                "emotion": emotion,
                "zone": zone,
                "camera_view": cv,
                "shot_id": shot["shot_id"]
            })

    # ---------------------------------------------------------
    # SHOT COMPOSITES
    # ---------------------------------------------------------
    IMAGE_SHOT_TYPES = [
        "closeup", "medium", "wide", "two_shot", "ots", "profile",
        "establishing", "tracking", "insert", "cutaway", "transition"
    ]

    for shot in shots:
        if shot["type"] in IMAGE_SHOT_TYPES:
            sc_alias = f"{scene_id}_SHOT_{shot['shot_id']}"

            # FIX: use the shot-specific backdrop instead of the base BG
            sb_alias = f"{scene_id}_ZB_SHOT_{shot['shot_id']}"

            deps = [sb_alias] + [
                f"{c['name'].upper()}_Sheet" for c in shot["characters"]
            ]

            graph["shot_composites"].append({
                "alias": sc_alias,
                "dependencies": deps,
                "description": shot["description"],
                "characters": [c["name"] for c in shot["characters"]],
                "camera_view": shot.get("camera_view", {}),
                "environment_zone": shot["environment_zone"],
                "shot_id": shot["shot_id"]
            })


    # ---------------------------------------------------------
    # DIALOG
    # ---------------------------------------------------------
    for shot in shots:
        dialog_lines = shot.get("dialog", [])
        if not dialog_lines:
            continue

        for i, line in enumerate(dialog_lines):
            speaker = line["speaker"].upper()
            d_alias = f"{scene_id}_D_{shot['shot_id']}_{i}"
            cu_alias = f"{scene_id}_CU_{shot['shot_id']}_{speaker}"

            graph["dialog"].append({
                "alias": d_alias,
                "dependencies": [f"{speaker}_Voice", cu_alias],
                "speaker": speaker,
                "shot_id": shot["shot_id"],
                "line_index": i
            })

    return graph


def generate_assets(registry, shots, graph):
    assets = []
    registry_map = {c["name"].upper(): c for c in registry["characters"]}

    # ---------------------------------------------------------
    # IDENTITY
    # ---------------------------------------------------------
    for item in graph["identity"]:
        name = item["character"]
        appearance = registry_map[name]["appearance_prompt"]

        if item["alias"].endswith("_Sheet"):
            instruction = f"create character sheet for {name}: {appearance}"
        else:
            instruction = f"design voice for {name}: {registry_map[name]['voice']}"

        assets.append({
            "alias": item["alias"],
            "alias_used": item["dependencies"],
            "instruction": instruction
        })

    # ---------------------------------------------------------
    # BACKGROUND
    # ---------------------------------------------------------
    bg = graph["background"]
    assets.append({
        "alias": bg["alias"],
        "alias_used": [],
        "instruction": f"create background for scene location: {bg['prompt']}"
    })

    # ---------------------------------------------------------
    # SHOT BACKDROPS
    # ---------------------------------------------------------
    for sb in graph["shot_backdrops"]:
        assets.append({
            "alias": sb["alias"],
            "alias_used": sb["dependencies"],
            "instruction": f"generate shot-specific backdrop: {sb['prompt']}"
        })

    # ---------------------------------------------------------
    # CLOSEUPS
    # ---------------------------------------------------------
    for cu in graph["closeups"]:
        name = cu["character"]
        emotion = cu["emotion"]
        zone = cu["zone"]

        shot = find_shot(shots, cu["shot_id"])

        char_cv = None
        for c in shot["characters"]:
            if c["name"].upper() == cu["character"]:
                char_cv = c.get("camera_view")
                break

        cv = char_cv or shot.get("camera_view", {})

        appearance = registry_map[name]["appearance_prompt"]

        cv_desc = (
            f"angle={cv.get('angle', 'front')}, "
            f"height={cv.get('height', 'eye-level')}, "
            f"distance={cv.get('distance', 'close')}, "
            f"framing={cv.get('framing', 'closeup')}, "
            f"facing={cv.get('facing', 'toward-character')}"
        )

        # FIX: use the shot-specific backdrop instead of the base BG
        sb_alias = f"{scene_id}_ZB_SHOT_{cu['shot_id']}"

        instruction = (
            f"closeup of {name}, showing {emotion.lower()} expression, "
            f"consistent with appearance: {appearance}. "
            f"Environment zone: {zone}. "
            f"Camera view: {cv_desc}. Tight framing."
        )

        assets.append({
            "alias": cu["alias"],
            "alias_used": [f"{name}_Sheet", sb_alias],
            "instruction": instruction
        })


    # ---------------------------------------------------------
    # SHOT COMPOSITES
    # ---------------------------------------------------------
    for sc in graph["shot_composites"]:
        chars = ", ".join(sc["characters"])

        shot = find_shot(shots, sc["shot_id"])
        shot_cv = shot.get("camera_view", {})

        char_views = []
        for c in shot["characters"]:
            cv = c.get("camera_view", {})
            char_views.append(
                f"{c['name']}: angle={cv.get('angle')}, "
                f"height={cv.get('height')}, "
                f"distance={cv.get('distance')}, "
                f"framing={cv.get('framing')}, "
                f"facing={cv.get('facing')}"
            )
        char_view_summary = "; ".join(char_views)

        instruction = (
            f"{sc['description']} "
            f"Environment zone: {sc['environment_zone']}. "
            f"Shot framing: angle={shot_cv.get('angle')}, height={shot_cv.get('height')}, "
            f"distance={shot_cv.get('distance')}, framing={shot_cv.get('framing')}, "
            f"facing={shot_cv.get('facing')}. "
            f"Per-character camera geometry: {char_view_summary}. "
            f"Characters visible: {chars}."
        )

        assets.append({
            "alias": sc["alias"],
            "alias_used": sc["dependencies"],
            "instruction": instruction
        })

    # ---------------------------------------------------------
    # DIALOG
    # ---------------------------------------------------------
    for d in graph["dialog"]:
        shot = find_shot(shots, d["shot_id"])
        line = shot["dialog"][d["line_index"]]["line"]
        appearance = registry_map[d["speaker"]]["appearance_prompt"]

        instruction = (
            f"{d['speaker']} speaks the line: \"{line}\". "
            f"Lip-sync and expression should match emotional context and "
            f"character appearance: {appearance}."
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
