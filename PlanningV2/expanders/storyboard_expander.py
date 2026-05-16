#!/usr/bin/env python3
"""
Expands short prompt → 6-panel cinematic storyboard for Qwen Image → Wan I2V/S2V pipeline.
Style-agnostic. Persistent character descriptors. Single-line output.
"""
import sys
import random
import re

def get_character_descriptors(user_prompt: str) -> dict:
    """
    Return persistent visual descriptors for temporal consistency.
    Uses your preferred attribute system: porcelain-flat skin, visual descriptors only, material-specific clothing.
    """
    # Base templates using your known preferences
    chars = {
        "protagonist": "young adult feminine-presenting individual, porcelain-flat skin tone #F5E6D3 zero gradients, almond eyes dark brown #5E483C monolids, straight dark-blonde hair low ponytail mid-back length, slim proportions 7.5 heads tall narrow shoulders long legs, wearing tattered dress matte polyester #1A2B3C frayed linen texture",
        "antagonist": "tall imposing figure, sharp angular facial structure high cheekbones low nasal bridge, deep-set eyes, long silver-streaked black hair straight texture, wearing heavy velvet robes #2C1B10 with brass chain accents matte metal texture, dramatic shadowed features bold ink outlines"
    }
    
    # Simple keyword overrides (extend with your material library)
    p = user_prompt.lower()
    if "knight" in p:
        chars["protagonist"] = "slim armored figure, porcelain-flat skin #F5E6D3, short-cropped dark-blonde hair, matte steel plate #4A5568 with worn leather #3E2723 accents, narrow shoulders long legs, determined expression soft smile"
    if "forest" in p or "spirit" in p:
        chars["antagonist"] = "ethereal humanoid, luminous pale skin #FDF6E3 porcelain-flat, almond eyes glowing amber #FFB74D, flowing translucent robes canvas texture #E8F4F8, barefoot, soft bioluminescent vein details #A5D6A7"
    
    return chars


def expand_storyboard(user_prompt: str, panel_count: int = 6, seed: int = None) -> str:
    if seed is not None:
        random.seed(seed)
    
    chars = get_character_descriptors(user_prompt)
    
    # Camera vocabulary (cinematic, Qwen Image handles spatial relationships well)
    cameras = [
        "wide establishing shot", "medium two-shot", "tight close-up on face",
        "low-angle hero shot", "dutch angle tension shot", "over-the-shoulder reverse",
        "extreme close-up on eyes", "high-angle vulnerability shot"
    ]
    
    # Lighting vocabulary (explicit, motivated sources)
    lighting = [
        "volumetric moonlight through broken window casting long shadows",
        "practical torchlight casting dynamic flickering shadows",
        "high-contrast chiaroscuro with cool rim light",
        "soft ambient fill with motivated warm key from side",
        "dramatic backlight silhouette with edge highlight",
        "cool ambient fill with warm practical accent",
        "hard directional light with deep contact shadows",
        "diffused overcast ambient with subtle bounce"
    ]
    
    # Location anchors (use your 60-30-10 palette concept implicitly)
    locations = [
        "crumbling stone dungeon moss-covered walls iron chains foreground",
        "vast abandoned castle hall shattered stained glass debris scattered",
        "narrow stone corridor flickering wall sconces deep perspective",
        "collapsed throne room dust motes in air broken columns",
        "underground chamber ancient runes glowing faint blue ambient",
        "balcony overlooking stormy courtyard lightning flashes background"
    ]
    
    # Dialogue pool (<8 words, for OmniVoice reference)
    dialogue = [
        "Why do this?", "There's still another way", "You don't understand",
        "Power requires sacrifice", "The past cannot be undone", "Your fear feeds me",
        "I remember who you were", "This ends now", "What do you want?",
        "Some chains are chosen"
    ]
    
    panels = []
    used_lines = set()
    
    for i in range(panel_count):
        cam = cameras[i % len(cameras)]
        light = lighting[i % len(lighting)]
        loc = locations[i % len(locations)]
        
        # Alternate focus for visual variety + consistency
        focus = chars["protagonist"] if i % 2 == 0 else chars["antagonist"]
        other = chars["antagonist"] if i % 2 == 0 else chars["protagonist"]
        
        # Positioning (Qwen Image excels at spatial relationships)
        positions = ["left third", "center frame", "right third", "foreground", "background", "off-screen"]
        pos = positions[i % len(positions)]
        other_pos = positions[(i + 3) % len(positions)]
        
        # Dialogue assignment (panels 2-5 for pacing)
        dialog_str = ""
        if 1 <= i <= 4:
            available = [d for d in dialogue if d not in used_lines]
            if available:
                line = random.choice(available)
                used_lines.add(line)
                speaker_label = "protagonist" if i % 2 == 0 else "antagonist"
                # Bubble anchor for OmniVoice sync reference (optional)
                bubble_anchor = "upper third near face" if "close-up" in cam else "upper frame"
                dialog_str = f" [OMNIVOICE REF: {speaker_label} says '{line}' bubble {bubble_anchor}]"
        
        # Assemble panel: visual descriptors first (Qwen prioritizes early tokens)
        panel = (
            f"Panel {i+1}: {cam}, {loc}, {light}. "
            f"{focus} positioned {pos}, {other} {other_pos}."
            f"{dialog_str}"
        ).strip()
        panels.append(panel)
    
    # Single-line assembly
    base = " | ".join(panels)
    # Optional negatives to avoid unwanted stylization
    suffix = "--no manga,anime,cel-shade,screentone,halftone,photorealism,3d-render"
    
    return f"{base} {suffix}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: storyboard_expander.py 'short prompt' [--seed N]", file=sys.stderr)
        sys.exit(1)
    
    seed = None
    args = sys.argv[1:]
    if "--seed" in args:
        idx = args.index("--seed")
        if idx + 1 < len(args):
            seed = int(args[idx + 1])
            args = args[:idx] + args[idx+2:]
    
    prompt = " ".join(args)
    print(expand_storyboard(prompt, seed=seed), end="")