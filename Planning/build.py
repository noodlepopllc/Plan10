#!/usr/bin/env python3
import json
import sys
import re
import os
from PIL import Image

# ============================================================================
# 🔧 IMPORT OR DEFINE GenerateZoneBackdrop
# If you have it in a separate module (e.g., backdrop_gen.py), uncomment:
# from backdrop_gen import GenerateZoneBackdrop
# ============================================================================
def GenerateZoneBackdrop(
    media: str,
    zone: str,
    output: str = "zone_backdrop.png",
    width: int = 1328,
    height: int = 1328,
    seed: int = -1,
    char_image: str = None,
):
    """Generate a repositioned zone backdrop from a master environment.
    Optionally bake a character into the plate using their reference PNG + metadata."""
    if not os.path.exists(media):
        raise FileNotFoundError(f"Environment source not found: {media}")

    # Handle video->frame if needed (your existing helper)
    background = video_to_img(media) if media.lower().endswith(('.mp4', '.mov')) else media

    images = [background]
    char_desc = ""

    # 🔑 CONDITIONAL CHARACTER INJECTION
    if char_image and os.path.exists(char_image):
        with Image.open(char_image) as img:
            images.append(char_image)  # Pass path, not PIL object, for EditImage compatibility
            raw_desc = img.info.get('Description', 'character')
            char_desc = (
                f"A single {raw_desc}. "
                "Preserve adult facial proportions, light cheekbone definition, and subtle jawline contour. "
                "Position naturally within the space, matching environmental lighting and perspective. "
            )

    # 🎥 AGGRESSIVE CAMERA REPOSITIONING PROMPT
    prompt_parts = [
        f"CHANGE CAMERA ANGLE: dolly/pan to frame {zone} inside this exact same room.",
        "COMPLETELY ERASE the original framing: remove all foreground crowds, subjects, and objects from the source view.",
        "Generate a NOVEL VIEWPOINT: shift perspective, change focal depth, reveal previously unseen architecture.",
        char_desc,
        "STRICTLY PRESERVE: lighting direction/intensity, wall & floor textures, color grading, "
        "architectural style, window/door placements, and overall atmosphere.",
        "SEAMLESSLY extend or reframe the scene to match the existing perspective.",
        "ALLOW: logical counters, registers, or fixtures that belong in this zone.",
        "NO text, NO style drift. Photorealistic cinematic environment shot."
    ]

    if char_image:
        prompt_parts.append("NO additional characters beyond the specified subject, NO foreground occlusions.")

    prompt = " ".join([p.strip() for p in prompt_parts if p.strip()])

    # 🎯 Route to your EditImage backend (adjust signature if needed)
    return EditImage(
        prompt=prompt,
        images=images,
        output=output,
        width=width,
        height=height,
        seed=seed
    )


def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_') or "asset"

# MOOD → DETERMINISTIC PHYSICAL REACTIONS (fallback only)
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

def _get_required_moods(sequence: dict, char_map: dict) -> set:
    """Scan sequence for exact (slug, mood) pairs needed."""
    ALLOWED_MOODS = {"neutral", "confident", "skeptical", "encouraging", "curious",
                     "supportive", "shy", "reassuring", "amused", "eager",
                     "nervous", "patient", "relieved", "stern", "defensive",
                     "frustrated", "overwhelmed", "playful"}
    required = set()
    for beat in sequence.get("beats", []):
        if beat.get("type") == "dialog":
            vis = beat.get("visible_chars", [])
            if vis and vis[0] in char_map:
                slug = char_map[vis[0]]["slug"]
                fa = beat.get("facial_action", "neutral")
                mood = fa.split(",")[0].strip().lower()
                if mood in ALLOWED_MOODS:
                    required.add((slug, mood))
    return required

def render_pipeline(registry_path: str, sequence_path: str) -> str:
    with open(registry_path) as f: registry = json.load(f)
    with open(sequence_path) as f: sequence = json.load(f)

    out = []
    env_slug = slugify(registry.get("environment_alias", "environment"))
    master_env_alias = f"bg_{env_slug}"

    # ========================================================================
    # PHASE 1: ASSETS (master env + zone backdrops + character sheets)
    # ========================================================================
    
    # 1a. Master environment (establishing shot)
    out.append(f'>> ALIAS: {master_env_alias}')
    out.append(f'create_background prompt="{registry["environment"]}" Height: 832, Width: 480, Seed: -1')

    # 1b. Per-character zone backdrops (with optional character baking)
    zone_backdrop_map = {}  # alias_slug -> zone_backdrop_alias
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        zone = c.get("background_zone", "center of the room")
        zone_slug = slugify(zone)[:20]  # Keep aliases CLI-safe
        zone_alias = f"{master_env_alias}_zone_{zone_slug}"
        zone_backdrop_map[slug] = zone_alias

        # Check if this character should be baked into their zone plate
        char_ref_path = f"assets/char_{slug}.png"  # Adjust to your asset structure
        should_bake = c.get("staged_character", False) and os.path.exists(char_ref_path)

        out.append(f'\n>> ALIAS: {zone_alias}')
        if should_bake:
            # Generate zone backdrop WITH character baked in (metadata + visual ref)
            out.append(f'generate_zone_backdrop media={master_env_alias}, zone="{zone}", char_image="{char_ref_path}", output={zone_alias}, Width: 1328, Height: 1328, Seed: -1')
        else:
            # Generate empty zone backdrop
            out.append(f'generate_zone_backdrop media={master_env_alias}, zone="{zone}", output={zone_alias}, Width: 1328, Height: 1328, Seed: -1')

    # 1c. Character reference sheets (for compositing)
    char_map = {}
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        char_map[c["id"]] = {"slug": slug, "design": f"design_{slug}", "voice": c["voice"]}
        out.append(f'\n>> ALIAS: char_{slug}')
        out.append(f'create_character_sheet prompt="{c["appearance_prompt"]}" Height: 832, Width: 480, Seed: -1')

    # ========================================================================
    # PHASE 2: REQUIRED CLOSEUPS (generate ONLY needed moods, predictable aliases)
    # ========================================================================
    required = _get_required_moods(sequence, char_map)
    for slug, mood in sorted(required):
        alias = f"compd_{slug}_{mood}"
        action = f"{mood}, mouth completely closed and still, lips sealed shut, zero lip motion, static facial expression, cropped at shoulders, NO hands, NO props"
        # 🔑 ROUTE TO CHARACTER'S ZONE BACKDROP FOR COMPOSITING
        backdrop = zone_backdrop_map.get(slug, master_env_alias)
        out.append(f'\n>> ALIAS: {alias}')
        out.append(f'composite_scene combining={backdrop}, char_{slug}, shot_type="closeup", action="{action}" Height: 832, Width: 480, Seed: -1')

    # ========================================================================
    # PHASE 3: VOICES (design)
    # ========================================================================
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        out.append(f'\n>> ALIAS: design_{slug}')
        out.append(f'design_voice voice="{c["voice"]}"')

    # ========================================================================
    # PHASE 4: DIALOG (3-Pass: Pregen Base → I2V Motion → S2V Lip-Sync)
    # ========================================================================
    dialog_idx = 1
    for beat in sequence["beats"]:
        if beat.get("type") != "dialog":
            continue
            
        visible_ids = beat.get("visible_chars", [])
        if not visible_ids or visible_ids[0] not in char_map:
            continue
            
        focus_cid = visible_ids[0]
        ch = char_map[focus_cid]
        slug = ch["slug"]
        
        raw_text = beat.get("text") or ""
        full_action = beat.get("facial_action", "neutral expression")
        
        # 🧹 Parse mood (first word) and motion (everything else)
        parts = full_action.split(",", 1)
        mood = parts[0].strip().lower()
        json_motion = "" #parts[1].strip() if len(parts) > 1 else ""

        # 🧹 Sanitize FIRST: move leaked [brackets] from text -> motion
        if raw_text:
            leaked = re.findall(r'\[(.*?)\]', raw_text)
            if leaked:
                raw_text = re.sub(r'\s*\[.*?\]\s*', ' ', raw_text).strip()
                json_motion = ", ".join(leaked) + (", " + json_motion if json_motion else "")

        # 🎯 Determine final motion (explicit JSON > mood fallback > default)
        if json_motion and len(json_motion) > 5:
            final_motion = json_motion
        else:
            fallback = MOOD_REACTIONS.get(mood, "subtle breathing, steady gaze")
            final_motion = fallback if isinstance(fallback, str) else fallback[dialog_idx % len(fallback)]

        # PASS 1: Base Headshot (predictable alias) - USE ZONE BACKDROP
        base_alias = f"compd_{slug}_{mood}"
        backdrop = zone_backdrop_map.get(slug, master_env_alias)
        
        # PASS 2: Motion Pass (I2V) - duration_sec as INT
        i2v_prompt = f"{mood}, {final_motion}, subtle camera drift, mouth completely closed and still, lips sealed shut, zero lip motion"
        i2v_alias = f"vid_motion_{dialog_idx:03d}"
        out.append(f'\n>> ALIAS: {i2v_alias}')
        out.append(f'image_to_video using={base_alias}, prompt="{i2v_prompt}", duration_sec=2 Height: 832, Width: 480, Seed: -1')

        # PASS 3: Lip-Sync Pass (S2V)
        if raw_text.strip():
            out.append(f'\n>> ALIAS: dialog_{dialog_idx:03d}')
            out.append(f'dialog_to_video using={i2v_alias}, audio={ch["design"]}, text="{raw_text}", prompt="lips moving naturally, preserve existing head motion, NO extra gestures, NO facial drift" Height: 832, Width: 480, Seed: -1')
        else:
            out.append(f'// Pure reaction: {i2v_alias} is final output')

        dialog_idx += 1

    return "\n".join(out)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python renderer.py registry.json sequence.json", file=sys.stderr)
        sys.exit(1)
    print(render_pipeline(sys.argv[1], sys.argv[2]))