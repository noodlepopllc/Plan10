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


def llm_rewrite(media: str | Path, prompt: str) -> str:
    """
    Deterministically rewrite a description into a short,
    purely physical, Qwen-friendly description.
    """
    output = llm_analyze_media(media, prompt)['analysis']
    return output.strip()


def describe_background_from_image(bg_path: Path) -> str:
    """
    Describe the background in purely physical terms.
    """
    return llm_rewrite(
        bg_path,
        "Describe the background in <Image> using only physical details "
        "(lighting, space, materials, geometry, style)."
    )


def describe_character_from_image(char_path: Path) -> str:
    """
    Describe the character in purely physical terms.
    """
    return llm_rewrite(
        char_path,
        "Describe the character in <Image> using only physical details: "
        "hair, skin tone, clothing, and general appearance. "
        "Do not mention personality or narrative."
    )


def rewrite_action_physical(action: str) -> str:
    """
    Rewrite the action into a purely physical description.
    """
    return llm_rewrite(
        "",
        f"Rewrite this action as a purely physical description appropriate for a camera shot. "
        f"Remove emotion and narrative. Focus on pose, facing direction, mouth, eyes, and simple movement:\n\n{action}"
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
) -> str:
    """
    Build the global Qwen-friendly prompt.
    """
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
    images = [background] + characters

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
    """
    Drop-in replacement for the old Flux compositor.
    Now uses the Qwen-native global synthesis compositor.
    """

    # Resolve seed
    if seed == -1:
        seed = random.randint(0, 100000)

    bg = Path(background_path)
    char_paths = [Path(c) for c in characters]

    # --- 1) Background description ---
    bg_desc = describe_background_from_image(bg)

    # --- 2) Character descriptions ---
    char_descs = [describe_character_from_image(c) for c in char_paths]

    # --- 3) Physical action rewrite ---
    action_physical = rewrite_action_physical(action)

    # --- 4) Camera description ---
    camera_desc = shot_type_to_camera_description(shot_type)

    # --- 5) Build global Qwen prompt ---
    prompt = build_qwen_prompt(
        bg_desc=bg_desc,
        char_descs=char_descs,
        camera_desc=camera_desc,
        action_physical=action_physical,
    )

    print(f"Compositing with prompt: {prompt}")

    # --- 6) Generate composite ---
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

    # 1) Background description
    bg_desc = describe_background_from_image(bg)

    # 2) Character descriptions
    char_descs = [describe_character_from_image(c) for c in chars]

    # 3) Physical action rewrite
    action_physical = rewrite_action_physical(action)

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
