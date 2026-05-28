import json
import re
import sys
from pathlib import Path
from collections import defaultdict

def resolve_location_description(registry):
    # assuming single location for now
    return registry["locations"][0]["description"]




# ---------------------------------------------------------
# CORE HELPERS
# ---------------------------------------------------------

def slugify(text):
    return re.sub(r'[^a-zA-Z0-9]+', '-', text).strip('-')


def find_shot(shots, shot_id):
    for s in shots:
        if s["shot_id"] == shot_id:
            return s
    raise KeyError(f"Shot {shot_id} not found")


def resolve_zone_description(registry, zone_name):
    for loc in registry["locations"]:
        for z in loc["zones"]:
            if z["zone_name"] == zone_name:
                return z["description"]

    return registry["locations"][0]["description"]



def assign_roles(shot):
    """
    Deterministic charA/charB assignment.
    First character = charA
    Second character = charB
    """
    roles = {}
    chars = shot["characters"]

    if len(chars) >= 1:
        roles[chars[0]["name"]] = "charA"
    if len(chars) >= 2:
        roles[chars[1]["name"]] = "charB"

    return roles


def select_variant_for_shot(shot, character_name_upper):
    """
    Decide which zone variant to use (Forward / ReverseA / ReverseB)
    based on layout + camera_side + solo/multi.
    """
    layout = shot.get("layout", "solo")
    camera_side = shot.get("camera_side", "center")
    chars = shot["characters"]

    # Multi-character → always Forward
    if len(chars) > 1:
        return "Forward"

    # Solo shots
    roles = assign_roles(shot)
    role = roles.get(character_name_upper.title(), None)  # names in roles are original case

    # If camera_side is explicit, use that
    if camera_side == "charA":
        return "ReverseA"
    if camera_side == "charB":
        return "ReverseB"

    # Fallback: role-based
    if role == "charA":
        return "ReverseA"
    if role == "charB":
        return "ReverseB"

    # Final fallback
    return "ReverseA"


# ---------------------------------------------------------
# DEPENDENCY GRAPH
# ---------------------------------------------------------

def build_dependency_graph(registry, scene_id, shots):
    graph = {
        "scene_id": scene_id,
        "identity": [],
        "background": None,
        "zone_backdrops": [],
        "base_composites": [],
        "shot_composites": [],
        "dialog": []
    }

    # ---------------- IDENTITY ----------------
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

    # ---------------- BACKGROUND ----------------
    location_prompt = resolve_location_description(registry)

    bg_alias = f"{scene_id}_BG"
    graph["background"] = {
        "alias": bg_alias,
        "dependencies": [],
        "prompt": location_prompt
    }

    # ---------------- ZONE VARIANTS ----------------
    zone_variants = {}  # zone_slug -> {variant_name: alias}

    for shot in shots:
        zone_slug = shot["environment_zone"]

        if zone_slug not in zone_variants:
            zone_variants[zone_slug] = {}

            for variant in ["Forward", "ReverseA", "ReverseB"]:
                alias = f"{scene_id}_ZV_{zone_slug}_{variant}"
                zone_variants[zone_slug][variant] = alias

                zone_name = shot["environment_zone"]
                zone_description = resolve_zone_description(registry, zone_name)


                if variant == "Forward":
                    prompt = (
                        f"{zone_description}\n\n"
                        f"FORWARD VIEW.\n"
                        f"Camera faces the primary subject direction.\n"
                        f"Show the environment exactly as described above, from the main camera orientation.\n"
                        "Show only the portion of the zone that is visible from this camera orientation."
                        "Do not duplicate or invent furniture."
                        f"Preserve all visible light sources and their positions."
                    )
                else:
                    prompt = (
                        f"REVERSE VIEW.\n"
                        "use the description of the asset to create a new zone 180 degree view of room  "
                        "keep lighting the same "
                        "no windows, 1 new piece of furniture if room is furnished that is appropriate to location"
                        f"preserve lighting on the environment.\n"
                    )


                graph["zone_backdrops"].append({
                    "alias": alias,
                    "dependencies": [bg_alias],
                    "zone_slug": zone_slug,
                    "variant": variant,
                    "prompt": prompt
                })

        shot["_zone_variants"] = zone_variants[zone_slug]

    # ---------------- BASE COMPOSITES ----------------
    for shot in shots:
        zone_slug = shot["environment_zone"]
        variants = shot["_zone_variants"]

        for char in shot["characters"]:
            name_upper = char["name"].upper()
            variant = select_variant_for_shot(shot, char["name"])
            zone_alias = variants[variant]

            base_alias = f"{scene_id}_BASE_{shot['shot_id']}_{name_upper}"

            graph["base_composites"].append({
                "alias": base_alias,
                "dependencies": [
                    f"{name_upper}_Sheet",
                    zone_alias
                ],
                "character": name_upper,
                "shot_id": shot["shot_id"],
                "zone_slug": zone_slug,
                "variant": variant
            })

    # ---------------- SHOT COMPOSITES ----------------
    for shot in shots:
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
            "shot_id": shot["shot_id"],
            "type": shot["type"]
        })

    # ---------------- DIALOG ----------------
    for shot in shots:
        dialog_lines = shot.get("dialog", [])
        if not dialog_lines:
            continue

        for i, line in enumerate(dialog_lines):
            speaker = line["speaker"].upper()
            d_alias = f"{scene_id}_D_{shot['shot_id']}_{i}"

            graph["dialog"].append({
                "alias": d_alias,
                "dependencies": [f"{speaker}_Voice", f"{scene_id}_SHOT_{shot['shot_id']}"],
                "speaker": speaker,
                "shot_id": shot["shot_id"],
                "line_index": i
            })

    return graph


# ---------------------------------------------------------
# ASSET GENERATION
# ---------------------------------------------------------

def generate_assets(registry, shots, graph):
    assets = []
    registry_map = {c["name"].upper(): c for c in registry["characters"]}
    # Precompute zone alias per shot_id from base_composites
    zone_alias_by_shot = {}
    for base in graph["base_composites"]:
        shot_id = base["shot_id"]
        _, zone_alias = base["dependencies"]
        zone_alias_by_shot.setdefault(shot_id, zone_alias)


    # ---------------- IDENTITY ----------------
    for item in graph["identity"]:
        name = item["character"]
        char = registry_map[name]

        if item["alias"].endswith("_Sheet"):
            base_prompt = char["appearance_prompt"]
            realism_block = (
                "shot on a full-frame DSLR, 50mm lens, natural skin texture, "
                "visible pores, subtle blemishes, micro-contrast, fine vellus hair, "
                "realistic subsurface scattering, no airbrushing, no smoothing, "
                "no painting, no illustration, no digital art, no concept art."
            )
            prompt = f"{base_prompt}. {realism_block}"
            instruction = f"create_character_sheet using prompt: {prompt}"
        else:
            voice_desc = char["voice"]
            instruction = f"design_voice using voice: {voice_desc}"

        assets.append({
            "alias": item["alias"],
            "alias_used": item["dependencies"],
            "instruction": instruction
        })

    # ---------------- BACKGROUND ----------------
    bg = graph["background"]
    bg_instruction = f"create_background using prompt: {bg['prompt']}"

    assets.append({
        "alias": bg["alias"],
        "alias_used": [],
        "instruction": bg_instruction
    })

    # ---------------- ZONE BACKDROPS ----------------
    for zb in graph["zone_backdrops"]:
        bg_alias = zb["dependencies"][0]
        instruction = (
            f"generate_backdrop using media: {bg_alias} asset, "
            "Camera distance: medium shot framing, "
            f"{zb['prompt']}"
        )


        assets.append({
            "alias": zb["alias"],
            "alias_used": zb["dependencies"],
            "instruction": instruction
        })

    # ---------------- BASE COMPOSITES ----------------
    for base in graph["base_composites"]:
        shot = find_shot(shots, base["shot_id"])
        character = base["character"]

        sheet_alias, zone_alias = base["dependencies"]

        char_data = next(c for c in shot["characters"] if c["name"].upper() == character)
        pose = char_data.get("pose") or "standing still"

        shot_type = shot["type"]

        instruction = (
            f"create a composite by combining {sheet_alias} and {zone_alias} assets, "
            f"shot type: {shot_type}, apply starting action: {pose}, "
            "integrate the pose naturally into the background using the asset descriptions "
            f"no camera transforms"
        )

        assets.append({
            "alias": base["alias"],
            "alias_used": [sheet_alias, zone_alias],
            "instruction": instruction
        })

    # ---------------- SHOT COMPOSITES ----------------
    for shot_node in graph["shot_composites"]:
        shot = find_shot(shots, shot_node["shot_id"])
        shot_type = shot["type"]

        action = ", ".join(
            (c.get("action") or c.get("pose") or "standing still")
            for c in shot["characters"]
        )

        shot_id = shot_node["shot_id"]
        zone_alias = zone_alias_by_shot[shot_id]
        char_sheets = [f"{c['name'].upper()}_Sheet" for c in shot["characters"]]

        deps = [zone_alias] + char_sheets

        instruction = (
            f"create a composite by combining {' and '.join(deps)} assets, "
            f"shot type: {shot_type}, apply action: {action}, "
            "integrate the character’s pose and action naturally into the background using the asset descriptions "
        )

        assets.append({
            "alias": shot_node["alias"],
            "alias_used": deps,
            "instruction": instruction
        })


    # ---------------- DIALOG ----------------
    for d in graph["dialog"]:
        shot = find_shot(shots, d["shot_id"])
        line = shot["dialog"][d["line_index"]]["line"]
        speaker = d["speaker"]

        sc_alias = f"{graph['scene_id']}_SHOT_{d['shot_id']}"
        audio_alias = f"{speaker}_Voice.wav"

        instruction = (
            f"create a dialog_to_video using {audio_alias} asset with {sc_alias} asset, "
            f"text: \"{line}\""
        )

        assets.append({
            "alias": d["alias"],
            "alias_used": d["dependencies"],
            "instruction": instruction
        })

    return assets


# ---------------------------------------------------------
# I2V GENERATION
# ---------------------------------------------------------

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

        micro_motion = ", ".join(
            (c.get("action") or c.get("pose") or "standing still")
            for c in shot["characters"]
        ) or "subtle environmental motion"

        instruction = (
            f"create an image_to_video using {asset['alias']} asset, "
            "integrate the character’s pose and action naturally into the background using the asset descriptions "
            f"action: {micro_motion}"
        )

        video_nodes.append({
            "alias": f"{scene_id}_VID_{shot_id}",
            "alias_used": [asset["alias"]],
            "instruction": instruction
        })

    return video_nodes


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == '__main__':
    basepath = sys.argv[1]
    scene_id = sys.argv[2]

    registry = json.loads(Path(f'{basepath}/registry.json').read_text())
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
