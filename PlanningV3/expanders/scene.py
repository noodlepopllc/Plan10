#!/usr/bin/env python3
import json, sys, re, os, math

sys.path.append('./lib')
from config import load_environ
load_environ()

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED","-1"))

# ------------------------------------------------------------
# SLUGIFY
# ------------------------------------------------------------
def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_') or "asset"

# ------------------------------------------------------------
# LOAD SINGLE SCENE FROM MULTI-SCENE REGISTRY
# ------------------------------------------------------------
def load_scene_registry(registry, scene_number):
    for scene in registry["scenes"]:
        if scene.get("scene_id") == scene_number:
            return {
                "scene_id": scene_number,
                "scene_alias": slugify(scene["scene_alias"]),
                "environment_alias": slugify(scene["environment_alias"]),
                "environment": scene["environment"],
                "zones": scene["zones"],
                "characters": registry["characters"]
            }
    raise ValueError(f"Scene {scene_number} not found")

# ------------------------------------------------------------
# MOOD TABLES (unchanged from your version)
# ------------------------------------------------------------
MOOD_REACTIONS = {
    "nervous": ["shifts weight", "averts gaze", "fidgets with hands"],
    "reassuring": ["leans forward slightly", "softens posture", "gentle smile"],
    "relieved": ["exhales visibly", "shoulders drop", "relaxes posture"],
    "playful": ["tilts head", "raises eyebrow", "light chuckle"],
    "frustrated": ["rubs temple", "crosses arms", "shakes head"],
    "confident": ["straightens spine", "holds steady gaze", "slight chin lift"],
    "skeptical": ["narrowed eyes", "slight head tilt", "crosses arms"],
    "encouraging": ["nods slowly", "leans in", "open palm gesture"],
    "curious": ["leans forward", "widens eyes slightly", "head cocked"],
    "supportive": ["places hand over heart", "softens expression", "nods"],
    "shy": "looks down, avoids eye contact",
    "amused": ["smirks", "shakes head lightly", "eyes crinkle"],
    "eager": ["bounces slightly", "leans in", "hands together"],
    "patient": ["steady breathing", "calm posture", "unblinking gaze"],
    "defensive": ["steps back", "crosses arms", "tightens jaw"],
    "overwhelmed": ["drops gaze", "exhales", "shoulders slump"],
    "stern": ["straightens posture", "locks jaw", "direct stare"],
    "neutral": ["subtle breathing", "steady gaze", "still posture"]
}

CLOSEUP_MOOD_VISUALS = {
    "nervous": "slightly widened eyes, tense jaw, subtle brow furrow, gaze slightly off-camera",
    "playful": "raised eyebrow, crinkled eye corners, slight head cock, amused direct gaze",
    "frustrated": "furrowed brow, tightened jaw, narrowed eyes, downward gaze",
    "skeptical": "narrowed eyes, raised eyebrow, slight head tilt, sidelong glance",
    "encouraging": "warm direct gaze, soft eye crinkle, relaxed forehead, slight forward lean posture",
    "curious": "widened eyes slightly, head cocked, focused direct gaze, relaxed mouth",
    "supportive": "soft eyes, gentle nod posture, relaxed brow, warm steady gaze",
    "shy": "downcast eyes, slight head duck, relaxed mouth, avoiding direct camera",
    "amused": "crinkled eye corners, slight closed-mouth smirk, raised eyebrow, light gaze",
    "eager": "bright wide eyes, alert expression, direct gaze, relaxed forehead",
    "defensive": "tight jaw, narrowed eyes, slight backward lean posture, guarded direct gaze",
    "overwhelmed": "dropped gaze, tense shoulders visible, furrowed brow, heavy eyelids",
    "stern": "direct unblinking gaze, firm jaw line, neutral brow, focused intense eyes",
    "reassuring": "HEAD TILTED SLIGHTLY FORWARD (5°), WARM DIRECT EYE CONTACT, SOFT BROW RELAXATION",
    "patient": "HEAD PERFECTLY LEVEL, EYES SOFT BUT NEUTRAL GAZE, JAW COMPLETELY LOOSE",
    "confident": "CHIN LIFTED 5°, SHOULDERS SQUARE AND BACK, DIRECT UNBLINKING GAZE",
    "neutral": "HEAD STRAIGHT, EYES LEVEL, BROWS FLAT, ZERO FACIAL TENSION"
}

# ------------------------------------------------------------
# EXTRACT REQUIRED MOODS FROM BEATS
# ------------------------------------------------------------
def get_required_moods(beats, char_slugs):
    required = set()
    for beat in beats:
        if beat.get("type") == "dialog":
            vis = beat.get("visible_chars", [])
            if not vis:
                continue
            char_name = vis[0]
            slug = slugify(char_name)
            mood = get_facial_mood(beat)
            required.add((slug, mood))
    return required

def get_facial_mood(beat):
    raw = beat.get("facial_action")
    if not raw or raw is None:
        return "neutral"
    return raw.split(",")[0].strip().lower()


def load_scene_beats(beats_json, scene_number):
    """
    Extracts beats for a specific scene from the multi-scene beats file.
    Expected format:
    {
        "scenes": [
            { "1": { "beats": [...] }},
            { "2": { "beats": [...] }},
            ...
        ]
    }
    """
    for scene_obj in beats_json["scenes"]:
        if str(scene_number) in scene_obj:
            return scene_obj[str(scene_number)]["beats"]

    raise ValueError(f"Scene {scene_number} not found in beats file")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage: python scene_builder.py registry.json scene_beats.json scene_number [--images-only] [--headshots-only]")
        sys.exit(1)

    registry_path = sys.argv[1]
    beats_path = sys.argv[2]
    scene_number = int(sys.argv[3])
    images_only = "--images-only" in sys.argv
    headshots_only = "--headshots-only" in sys.argv

    with open(registry_path) as f: registry = json.load(f)
    with open(beats_path) as f: beats_data = json.load(f)

    reg = load_scene_registry(registry, scene_number)
    beats = load_scene_beats(beats_data, scene_number)


    out = []

    # ------------------------------------------------------------
    # SCENE-SCOPED SLUGS
    # ------------------------------------------------------------
    scene_slug = f"scene{reg['scene_id']}_{reg['scene_alias']}"
    env_alias = f"bg_{scene_slug}"
    env_prompt = reg["environment"]

    # ------------------------------------------------------------
    # 1. MASTER BACKGROUND
    # ------------------------------------------------------------
    out.append(f'\n>> ALIAS: {env_alias}')
    out.append(f'create_background prompt="{env_prompt}" Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}')

    # ------------------------------------------------------------
    # 2. ZONES PER CHARACTER
    # ------------------------------------------------------------
    zone_map = {}
    for zone in reg["zones"]:
        char = next(c for c in reg["characters"] if c["id"] == zone["character_id"])
        char_slug = slugify(char["alias_slug"])
        zone_slug = slugify(zone["background_zone"])[:20]

        alias = f"bd_{scene_slug}_{char_slug}_{zone_slug}"
        zone_map[char_slug] = alias

        out.append(f'\n>> ALIAS: {alias}')
        out.append(
            f'generate_backdrop media={env_alias}, zone="{zone["background_zone"]}", '
            f'master_prompt="{env_prompt}", output={alias}, Width: 1328, Height: 1328, Seed: {SEED}'
        )

    # ------------------------------------------------------------
    # 3. HEADSHOTS (dialog moods)
    # ------------------------------------------------------------
    char_slugs = [slugify(c["alias_slug"]) for c in reg["characters"]]
    required = get_required_moods(beats, char_slugs)

    for slug, mood in sorted(required):
        alias = f"compd_{scene_slug}_{slug}_{mood}"
        cues = CLOSEUP_MOOD_VISUALS.get(mood, CLOSEUP_MOOD_VISUALS["neutral"])
        backdrop = zone_map.get(slug, env_alias)

        out.append(f'\n>> ALIAS: {alias}')
        out.append(
            f'composite_scene combining={backdrop}, char_{slug}, shot_type="closeup", '
            f'action="{cues}, mouth closed, no props" Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}'
        )

    if headshots_only:
        print("\n".join(out))
        return

    # ------------------------------------------------------------
    # 4. PROCESS BEATS (dialog + action)
    # ------------------------------------------------------------
    idx = 1
    for beat in beats:
        btype = beat.get("type")

        # -------------------------
        # DIALOG
        # -------------------------
        if btype == "dialog":
            char_name = beat["visible_chars"][0]
            slug = slugify(char_name)
            mood = beat.get("facial_action", "neutral").split(",")[0].strip().lower()
            base_alias = f"compd_{scene_slug}_{slug}_{mood}"

            text = beat.get("text", "")
            alias = f"dialog_{scene_slug}_{idx:03d}"

            out.append(f'\n>> ALIAS: {alias}')
            out.append(
                f'dialog_to_video using={base_alias}, audio=design_{slug}, text="{text}", '
                f'prompt="natural lip sync, no head motion" Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}'
            )

            if not images_only:
                motion_alias = f"vid_{scene_slug}_{idx:03d}"
                out.append(f'\n>> ALIAS: {motion_alias}')
                out.append(
                    f'image_to_video using={base_alias}, prompt="{mood}, subtle motion", '
                    f'duration_sec=2 Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}'
                )

        # -------------------------
        # ACTION
        # -------------------------
        if btype == "action":
            visible = beat.get("visible_chars", [])
            if not visible:
                continue

            char_slugs = [slugify(v) for v in visible]
            shot = beat.get("shot_type", "medium")
            motion = beat.get("motion_prompt", "")
            facial = get_facial_mood(beat)

            focus_slug = char_slugs[0]
            backdrop = zone_map.get(focus_slug, env_alias)

            alias = f"action_{scene_slug}_{idx:03d}"
            action_prompt = f"{facial}, {motion}".strip(", ")

            out.append(f'\n>> ALIAS: {alias}')
            out.append(
                f'composite_scene combining={backdrop}, '
                f'{", ".join("char_"+s for s in char_slugs)}, '
                f'shot_type="{shot}", action="{action_prompt}" '
                f'Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}'
            )

            if motion and not images_only:
                dur = math.ceil(float(beat.get("duration", 3.0)))
                motion_alias = f"vid_action_{scene_slug}_{idx:03d}"
                out.append(f'\n>> ALIAS: {motion_alias}')
                out.append(
                    f'image_to_video using={alias}, prompt="{motion}", '
                    f'duration_sec={dur} Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}'
                )

        idx += 1

    print("\n".join(out))


if __name__ == "__main__":
    main()
