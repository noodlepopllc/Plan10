import argparse, os, random
from pathlib import Path
from typing import List, Optional

from plan10.lib.config import load_config
load_config()

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))

SEED = int(os.environ.get("SEED", "-1"))
SEED = random.randint(0, 100000) if SEED == -1 else SEED

from plan10.lib.qwen_llm import llm_analyze_media
from plan10.lib.image_edit import ImageEditQwen2

def truncate(text: str, max_chars: int = 240) -> str:
    return text[:max_chars].rsplit(" ", 1)[0]

def llm_rewrite(media: str | Path, prompt: str) -> str:
    """
    Deterministically rewrite a description into a short,
    purely physical, Qwen-friendly description.
    """
    output = llm_analyze_media(str(media), prompt)['analysis']
    print(f'MEDIA: {media}\n PROMPT: "{prompt}"\n OUTPUT: "{output.strip()}"\n')
    return output.strip()

def extract_mood(text: str) -> str:
    """
    Extract a simple mood label from an action or expression string.
    Returns a single word like: angry, sad, happy, fearful, disgusted, surprised, neutral.
    """
    output = llm_rewrite(
        "",
        f"""
You are a deterministic mood classifier.

Task:
- Read the following description (action or expression).
- Classify the dominant emotional mood.
- Respond with ONE word from this set:
  angry, sad, happy, fearful, disgusted, surprised, neutral.

Description:
{text}
"""
    )
    return output.strip().lower()

def mood_to_facial_description(mood: str) -> str:
    """
    Map a mood label into a physical facial description.
    No emotion words, only anatomy: brows, eyes, mouth, jaw, head angle.
    """
    return llm_rewrite(
        "",
        f"""
You are a deterministic facial-expression mapper.

Task:
- Convert the given mood into a physical facial description.
- Describe only brows, eyes, mouth, jaw, and head angle.
- No emotion words, no narrative, no backstory.
- 1–2 short sentences.

Mood:
{mood}
"""
    )

def determine_mood(action: Optional[str], expression: Optional[str]) -> str:
    """
    Priority:
    1) If action exists, mood comes from action.
    2) Else if expression exists, mood comes from expression.
    3) Else, neutral.
    """
    if action:
        return extract_mood(action)
    if expression:
        return extract_mood(expression)
    return "neutral"


def describe_background_from_image(bg_path: Path) -> str:
    """
    Describe the background in purely physical terms.
    """
    return llm_rewrite(
        bg_path,'''
Describe the background in <Image> using only physical details.
Limit the description to 1–2 short sentences.
Avoid listing objects individually.
Avoid narrative or interpretation.'''
    )


def describe_character_from_image(char_path: Path) -> str:
    """
    Describe the character in purely physical terms.
    """
    return llm_rewrite(
        char_path,
'''The image is a character sheet. Describe only the character’s identity and appearance.
Limit to 1–2 short sentences.
Do NOT describe poses, front/back views, multiple angles, or any specific stance.
Do NOT mention that the sheet shows different views.
Focus only on stable identity traits: hair, skin tone, outfit, and overall style.
'''
    )


def rewrite_action_physical(action: str) -> str:
    return llm_rewrite(
        "",
        f"""
Rewrite the following action as a purely physical description.
Limit to 1–2 short sentences.
Remove emotion, narrative, and causal logic.
Describe only pose, facing direction, and simple movement.

Action:
{action}
"""
    )



def shot_type_to_camera_description(shot_type: str) -> str:
    """
    Map shot type to a camera description.
    """
    shot_type = shot_type.lower()

    if shot_type == "closeup":
        return "tight closeup, head and shoulders, static camera, eye-level"

    if shot_type == "medium":
        return "medium shot, waist-up, static camera, eye-level"

    if shot_type == "two_shot":
        return "two-shot, both characters visible, waist-up, static camera, eye-level"

    if shot_type == "ots":
        return "over-the-shoulder framing, foreground shoulder visible, focus on the other character"

    if shot_type == "profile_left":
        return "profile view, character facing left, side of face visible"

    if shot_type == "profile_right":
        return "profile view, character facing right, side of face visible"

    return f"{shot_type} shot, static camera, eye-level"
    


def build_qwen_prompt(
    bg_desc: str,
    char_descs: List[str],
    camera_desc: str,
    action_physical: str,
    face_descs: Optional[List[str]] = None,
) -> str:
    lines = []
    lines.append("A composite scene using the provided background and character reference images.")
    lines.append("")
    lines.append("BACKGROUND:")
    lines.append(f"- {bg_desc}")
    lines.append("")
    lines.append("SHOT:")
    lines.append(f"- {camera_desc}")
    lines.append("")
    lines.append("CHARACTERS:")
    for i, desc in enumerate(char_descs, start=1):
        lines.append(f"{i}. {desc}")
        if face_descs and i <= len(face_descs) and face_descs[i-1]:
            lines.append(f"   FACIAL EXPRESSION: {face_descs[i-1]}")
    lines.append("")
    lines.append("ACTION:")
    lines.append(f"- {action_physical}")
    lines.append("")
    lines.append("LIGHTING:")
    lines.append("- Match the background lighting.")
    lines.append("")
    lines.append("STYLE:")
    lines.append("- Realistic.")

    return "\n".join(lines)



def qwen_generate(
    background: Path,
    characters: List[Path],
    prompt: str,
    output: Path,
    width: int = WIDTH,
    height: int = HEIGHT,
    seed: int = SEED,
):
    """
    Call Qwen-Image-2.1 using ImageEditQwen2 in multi-image mode.
    """
    editor = ImageEditQwen2()

    # IMPORTANT: multi-image synthesis mode = pass all images at once
    images = [str(x) for x in [background] + characters]

    status = editor.generate(
        prompt=prompt,
        images=images,
        output=str(output),
        width=width,
        height=height,
        seed=seed
    )

    return status

def CompositeSceneQwen(
    background_path: str,
    characters: list[str],
    shot_type: str = "medium",
    action: str = "hair swaying gently",
    output: str = "composite.png",
    seed: int = -1,
    width: int = WIDTH,
    height: int = HEIGHT,
):
    # seed, paths...
    bg = Path(background_path)
    char_paths = [Path(c) for c in characters]

    bg_desc = truncate(describe_background_from_image(bg))
    char_descs = [truncate(describe_character_from_image(c)) for c in char_paths]
    action_physical = truncate(rewrite_action_physical(action))
    camera_desc = shot_type_to_camera_description(shot_type)

    # mood from action (for now)
    mood = extract_mood(action)

    # facial descriptions per character
    face_descs: List[str] = []

    if shot_type == "ots" and len(char_paths) == 2:
        # char 1 foreground → no face
        face_descs.append("")  # or None
        # char 2 background → face described
        face_descs.append(truncate(mood_to_facial_description(mood)))
    else:
        # all characters get a face description
        for _ in char_paths:
            face_descs.append(truncate(mood_to_facial_description(mood)))

    prompt = build_qwen_prompt(
        bg_desc=bg_desc,
        char_descs=char_descs,
        camera_desc=camera_desc,
        action_physical=action_physical,
        face_descs=face_descs,
    )

    print(f"Compositing with prompt: {prompt}")

    status = qwen_generate(
        background=bg,
        characters=char_paths,
        prompt=prompt,
        output=Path(output),
        width=width,
        height=height,
        seed=seed,
    )

    return status



def main():
    parser = argparse.ArgumentParser(description="Qwen-native compositor")
    parser.add_argument("-B", "--background", type=Path, required=True)
    parser.add_argument("-C", "--character", type=Path, action="append", default=[])
    parser.add_argument("-S", "--shot-type", type=str, required=True)
    parser.add_argument("-A", "--action", type=str, required=True)
    parser.add_argument("-O", "--output", type=Path, required=True)
    parser.add_argument("-Z", "--bg-transform", type=str, default=None)
    parser.add_argument("-R", "--raw", action="store_true")

    args = parser.parse_args()

    bg = args.background
    chars = args.character
    shot_type = args.shot_type
    action = args.action
    out_path = args.output

    if not bg.exists():
        raise FileNotFoundError(f"Background not found: {bg}")

    for c in chars:
        if not c.exists():
            raise FileNotFoundError(f"Character ref not found: {c}")

    # --- 1) Background description ---
    bg_desc = truncate(describe_background_from_image(bg))

    # --- 2) Character descriptions ---
    char_descs = [truncate(describe_character_from_image(c)) for c in chars]

    # --- 3) Physical action rewrite ---
    action_physical = truncate(rewrite_action_physical(action))

    # 4) Camera description
    camera_desc = shot_type_to_camera_description(shot_type)

    # 5) Build global prompt
    prompt = build_qwen_prompt(
        bg_desc=bg_desc,
        char_descs=char_descs,
        camera_desc=camera_desc,
        action_physical=action_physical,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 6) Generate composite
    status = qwen_generate(
        background=bg,
        characters=chars,
        prompt=prompt,
        output=out_path
    )
    print(status)

if __name__ == "__main__":
    main()
