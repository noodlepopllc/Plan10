#!/usr/bin/env python3
import json, sys, re, os

sys.path.append('./lib')
from config import load_environ
load_environ()

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED","-1"))

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

# ADD THIS BELOW MOOD_REACTIONS
CLOSEUP_MOOD_VISUALS = {
    "nervous": "slightly widened eyes, tense jaw, subtle brow furrow, gaze slightly off-camera",
    "reassuring": "soft eyes, relaxed forehead, warm direct gaze, gentle downward head tilt",
    "relieved": "relaxed brow, soft eyelids, gentle exhale posture, calm downward-to-direct gaze",
    "playful": "raised eyebrow, crinkled eye corners, slight head cock, amused direct gaze",
    "frustrated": "furrowed brow, tightened jaw, narrowed eyes, downward gaze",
    "confident": "steady direct gaze, relaxed jaw, slight chin lift, calm unblinking eyes",
    "skeptical": "narrowed eyes, raised eyebrow, slight head tilt, sidelong glance",
    "encouraging": "warm direct gaze, soft eye crinkle, relaxed forehead, slight forward lean posture",
    "curious": "widened eyes slightly, head cocked, focused direct gaze, relaxed mouth",
    "supportive": "soft eyes, gentle nod posture, relaxed brow, warm steady gaze",
    "shy": "downcast eyes, slight head duck, relaxed mouth, avoiding direct camera",
    "amused": "crinkled eye corners, slight closed-mouth smirk, raised eyebrow, light gaze",
    "eager": "bright wide eyes, alert expression, direct gaze, relaxed forehead",
    "patient": "calm steady gaze, relaxed facial muscles, neutral brow, soft focus",
    "defensive": "tight jaw, narrowed eyes, slight backward lean posture, guarded direct gaze",
    "overwhelmed": "dropped gaze, tense shoulders visible, furrowed brow, heavy eyelids",
    "stern": "direct unblinking gaze, firm jaw line, neutral brow, focused intense eyes",
    "neutral": "relaxed facial muscles, direct soft gaze, even breathing posture, calm expression"
}

def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_') or "asset"

def _get_required_moods(sequence: dict, char_slugs: list) -> set:
    ALLOWED_MOODS = {"neutral", "confident", "skeptical", "encouraging", "curious",
                     "supportive", "shy", "reassuring", "amused", "eager",
                     "nervous", "patient", "relieved", "stern", "defensive",
                     "frustrated", "overwhelmed", "playful"}
    required = set()
    for beat in sequence.get("beats", []):
        if beat.get("type") == "dialog":
            vis = beat.get("visible_chars", [])
            if vis:
                idx = vis[0] - 1
                if idx < len(char_slugs):
                    slug = char_slugs[idx]
                    fa = beat.get("facial_action", "neutral")
                    mood = fa.split(",")[0].strip().lower()
                    if mood in ALLOWED_MOODS:
                        required.add((slug, mood))
    return required

def main():
    if len(sys.argv) < 3:
        print("Usage: python dialog.py registry.json dialog.json [char_slugs...] [--headshots-only] [--images-only]", file=sys.stderr)
        sys.exit(1)
    
    registry_path = sys.argv[1]
    dialog_path = sys.argv[2]
    
    headshots_only = "--headshots-only" in sys.argv
    images_only = "--images-only" in sys.argv
    char_slugs = [a for a in sys.argv[3:] if not a.startswith("--")]
    
    with open(registry_path) as f: registry = json.load(f)
    with open(dialog_path) as f: dialog_seq = json.load(f)
    
    out = []
    env_slug = slugify(registry.get("environment_alias", "environment"))
    master_env_alias = f"bg_{env_slug}"
    
    # Build zone map
    zone_backdrop_map = {}
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        zone = c.get("background_zone", "center of the room")
        zone_alias = f"bd_{env_slug}_zone_{slugify(zone)[:20]}"
        zone_backdrop_map[slug] = zone_alias
    
    # Build char_map for lookup
    char_map = {}
    for i, c in enumerate(registry["characters"]):
        slug = slugify(c.get("alias_slug", c["name"]))
        char_map[i+1] = {"slug": slug, "design": f"design_{slug}"}
    
    # ========================================================================
    # PHASE 1: HEADSHOTS (ALWAYS GENERATED)
    # ========================================================================
    required = _get_required_moods(dialog_seq, [c["alias_slug"] for c in registry["characters"]])
    for slug, mood in sorted(required):
        alias = f"compd_{slug}_{mood}"
        action = f"{mood_cues}, mouth closed and relaxed, lips gently together, cropped at shoulders, NO hands, NO props, NO motion blur"
        backdrop = zone_backdrop_map.get(slug, master_env_alias)
        
        out.append(f'\n>> ALIAS: {alias}')
        out.append(f'composite_scene combining={backdrop}, char_{slug}, shot_type="closeup", action="{action}" Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}')

    for slug, mood in sorted(required):
        alias = f"compd_{slug}_{mood}_medium"
        action = f"{mood_cues}"
        backdrop = zone_backdrop_map.get(slug, master_env_alias)
        
        out.append(f'\n>> ALIAS: {alias}')
        out.append(f'composite_scene combining={backdrop}, char_{slug}, shot_type="medium", action="{action}" Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}')
    
    # Exit early if just previewing headshots
    if headshots_only:
        print("\n".join(out))
        return
    
    # ========================================================================
    # PHASE 2: DIALOG & LIP-SYNC
    # ========================================================================
    dialog_idx = 1
    for beat in dialog_seq.get("beats", []):
        if beat.get("type") != "dialog": continue
        visible_ids = beat.get("visible_chars", [])
        if not visible_ids or visible_ids[0] not in char_map: continue
        
        focus_cid = visible_ids[0]
        ch = char_map[focus_cid]
        slug = ch["slug"]
        raw_text = beat.get("text") or ""
        full_action = beat.get("facial_action", "neutral expression")
        
        # Parse mood/motion
        parts = full_action.split(",", 1)
        mood = parts[0].strip().lower()
        json_motion = ""
        if raw_text:
            leaked = re.findall(r'\[(.*?)\]', raw_text)
            if leaked:
                raw_text = re.sub(r'\s*\[.*?\]\s*', ' ', raw_text).strip()
                json_motion = ", ".join(leaked) + (", " + json_motion if json_motion else "")
        
        final_motion = json_motion if json_motion and len(json_motion) > 5 else MOOD_REACTIONS.get(mood, "subtle breathing, steady gaze")
        if isinstance(final_motion, list):
            final_motion = final_motion[dialog_idx % len(final_motion)]
        
        base_alias = f"compd_{slug}_{mood}"
        
        # STEP 1: Static S2V (always runs if not headshots_only)
        static_alias = f"dialog_static_{dialog_idx:03d}"
        out.append(f'\n>> ALIAS: {static_alias}')
        out.append(f'dialog_to_video using={base_alias}, audio={ch["design"]}, text="{raw_text}", prompt="lips moving naturally, preserve facial structure, NO head motion, NO background drift, mouth articulation matches audio phonemes" Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}')
        
        # STEP 2: I2V + Motion S2V (skipped if --images-only)
        if not images_only:
            motion_alias = f"vid_motion_{dialog_idx:03d}"
            i2v_prompt = f"{mood}, {final_motion}, subtle camera drift, mouth completely closed and still, lips sealed shut, zero lip motion"
            out.append(f'\n>> ALIAS: {motion_alias}')
            out.append(f'image_to_video using={base_alias}_medium, prompt="{i2v_prompt}", duration_sec=2 Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}')
        
        dialog_idx += 1
    
    print("\n".join(out))

if __name__ == "__main__":
    main()