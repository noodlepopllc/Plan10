from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import gc
import torch
import os, random, json

from plan10.lib.config import load_environ
load_environ()

from plan10.lib.util import load_metadata
from plan10.lib.image_analysis import AnalyzeImage
from plan10.lib.qwen_llm import llm_analyze_media
from PIL import Image

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import huggingface_hub
import gc
from rembg import remove


WH_RATIO_TO_SIZE = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400),
    "3:2": (2528, 1696), "2:3": (1696, 2528), "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

WH_RATIO_TO_SIZE_SANE = {
    "1:1": (1024, 1024), "4:3": (1152, 864), "3:4": (864, 1152),
    "3:2": (1216, 832), "2:3": (832, 1216), "16:9": (1344, 768),
    "9:16": (768, 1344),
}

WH_RATIO_TO_SIZE_LOW_VRAM = {
    "1:1": (768, 768), "4:3": (896, 672), "3:4": (672, 896),
    "3:2": (960, 640), "2:3": (640, 960), "16:9": (1024, 576),
    "9:16": (576, 1024),
}

def normalize_to_target_resolution(image_path, target_width, target_height):
    """
    Pads the image to match the target aspect ratio FIRST, 
    then resizes to the exact target resolution. 
    This prevents the character from being shrunk and distorted.
    """
    img = Image.open(image_path).convert("RGBA")
    
    target_ratio = target_width / target_height
    img_ratio = img.width / img.height
    
    if img_ratio > target_ratio:
        # Image is wider than target: Pad TOP and BOTTOM
        new_height = int(img.width / target_ratio)
        canvas = Image.new("RGBA", (img.width, new_height), (0, 0, 0, 0))
        offset_y = (new_height - img.height) // 2
        canvas.paste(img, (0, offset_y))
    else:
        # Image is taller than target: Pad LEFT and RIGHT
        new_width = int(img.height * target_ratio)
        canvas = Image.new("RGBA", (new_width, img.height), (0, 0, 0, 0))
        offset_x = (new_width - img.width) // 2
        canvas.paste(img, (offset_x, 0))
    
    # Now resize the perfectly padded canvas to the exact target resolution
    final_img = canvas.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Save normalized version
    normalized_path = image_path.replace(".png", f"_norm_{target_width}x{target_height}.png")
    final_img.save(normalized_path)
    return normalized_path

def expand_prompt_with_qwen_image(user_prompt: str) -> dict:
    """Load Qwen-Image-2.1 prompt rewriter, expand prompt, then fully unload."""
    
    model_id = "Qwen/Qwen-Image-2.1-PE-T2I"
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    ).eval()
    
    # Load system prompt
    sys_prompt_path = huggingface_hub.hf_hub_download(model_id, "system_prompt.txt")
    with open(sys_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    # Format input
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=16256,
            do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
        )
    
    gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Parse result
    thinking, _, answer = gen.partition("</think>")
    result = json.loads(answer.strip())
    
    # Unload and clean up
    del model
    del tokenizer
    del inputs
    del out
    del text
    del gen
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

import json
import gc
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import huggingface_hub

def expand_edit_prompt_with_qwen_image(image_paths: list, user_prompt: str) -> dict:
    """Load Qwen-Image-2.1 I2I prompt rewriter, expand edit prompt, then fully unload."""
    
    if len(image_paths) > 10:
        raise ValueError(f"Qwen-Image-2.1 I2I supports maximum 10 images, got {len(image_paths)}")
    
    model_id = "Qwen/Qwen-Image-2.1-PE-I2I"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    ).eval()
    
    # Load system prompt
    sys_prompt_path = huggingface_hub.hf_hub_download(model_id, "system_prompt.txt")
    with open(sys_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # Load all input images (resize for speed)
    input_images = [prepare_image_for_expander(path, max_size=512) for path in image_paths]
    
    # Build user content with all images
    user_content = []
    for img in input_images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": user_prompt})
    
    # Format input
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=True,
    ).to(model.device)
    
    # Generate - increased to 8192 to handle long rewritten_prompt fields
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=24000,  # Increased from 4096
            do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
        )
    
    gen = processor.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    
    # Split thinking from the answer
    thinking, _, answer = gen.partition("</think>")
    answer = answer.strip()
    
    # Try to parse JSON with fallback for truncated output
    try:
        result = json.loads(answer)
    except json.JSONDecodeError as e:
        print(f"[Warning] JSON parse failed: {e}")
        print(f"[Warning] Attempting to salvage truncated JSON...")
        
        # Try to find and close the rewritten_prompt string
        if '"rewritten_prompt":' in answer:
            # Find where the prompt value starts
            start_idx = answer.find('"rewritten_prompt":') + len('"rewritten_prompt":')
            # Find the last complete sentence (ends with period)
            prompt_text = answer[start_idx:].strip()
            if prompt_text.startswith('"'):
                prompt_text = prompt_text[1:]  # Remove opening quote
            
            # Find last period and truncate there
            last_period = prompt_text.rfind('.')
            if last_period > 0:
                prompt_text = prompt_text[:last_period + 1]
            
            # Reconstruct valid JSON
            salvaged = f'{{"rewritten_prompt": "{prompt_text}", "wh_ratio": "", "ratio_follow": "<image1>"}}'
            try:
                result = json.loads(salvaged)
                print(f"[Warning] Successfully salvaged JSON with truncated prompt")
            except:
                # Final fallback - just use the raw text
                result = {
                    "rewritten_prompt": prompt_text,
                    "wh_ratio": "",
                    "ratio_follow": "<image1>"
                }
        else:
            # No rewritten_prompt found at all - use raw text
            result = {
                "rewritten_prompt": answer,
                "wh_ratio": "",
                "ratio_follow": "<image1>"
            }
    
    # Unload and clean up
    del model
    del processor
    del inputs
    del out
    del input_images
    del gen
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

'''
# Usage - single image
result = expand_edit_prompt_with_qwen_image(["input.png"], "make the sky sunset")

# Usage - multiple images (up to 10)
result = expand_edit_prompt_with_qwen_image(
    ["char1.png", "char2.png", "background.png"],
    "combine these characters in the background scene"
)
print(result)
# {"rewritten_prompt": "...", "wh_ratio": "", "ratio_follow": "<image1>"}

# Usage
result = expand_prompt_with_qwen_image("一只在雨中弹吉他的柯基")
print(result)
# {"rewritten_prompt": "<long detailed English prompt>", "wh_ratio": "16:9"}
'''

def prompt_metadata(imgpath, prompt=''):
    from plan10.lib.util import wait_for_file
    wait_for_file(imgpath)
    if prompt:
        with Image.open(imgpath) as target_image:
            metadata = load_metadata(target_image)
            target_image.load()
        metadata.add_text("GenerationPrompt", prompt)
        target_image.save(imgpath, pnginfo=metadata)
        return prompt
    else:
        with Image.open(imgpath) as target_image:
            return target_image.info.get("GenerationPrompt", "")

class ImageGenQwen21(object):
    def __init__(self,vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
        self.vrlimit = vrlimit
        self.pipe = None

    def __enter__(self):
        if not self.pipe:
            vram_config = {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": "cpu",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda"
            }
            self.pipe = QwenImage21Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors", **vram_config),
                ],
                processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
                        vram_limit=self.vrlimit,
                )

    def generate(self, prompt, output, width, height, seed):
        if not self.pipe:
            self.__enter__()
        image = self.pipe(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_inference_steps=40,
                tiled=(width > 1536) or (height > 1536),
                tile_size = 384,
                tile_stride = 320
            )
        image.save(output)
        return {"status":"success", "output_path":output}

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        gc.collect()
        if torch.cuda.is_available():  # ✅ Was `if torch.cuda:` (always truthy)
            torch.cuda.empty_cache()

class ImageEditQwen21(object):
    def __init__(self,vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
        self.vrlimit = vrlimit
        self.pipe = None

    def get_pipe(self):
        if not self.pipe:
            self.__enter__()
        return self.pipe
    
    def __enter__(self):
        if not self.pipe:
            vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cpu",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda"
            }
            self.pipe = QwenImage21Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors", **vram_config),
                ],
                processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
                            vram_limit=self.vrlimit
                )
        return self

    def generate(self, prompt, images, output, width, height, seed):
        if not self.pipe:
            self.__enter__()
        # Safely handle empty/character-only lists
        edit_images = []
        for item in images:
            if isinstance(item, Image.Image):
                # Already a PIL image → use directly
                edit_images.append(item.convert("RGB"))
            elif isinstance(item, str):
                # File path → load it
                edit_images.append(Image.open(item).convert("RGB"))
            else:
                raise TypeError(f"Unsupported image type: {type(item)}")
        if seed == -1: seed = random.randint(0, 1000000)

        image = self.pipe(
            prompt, edit_image=edit_images, seed=seed,
            height=height, width=width
        )
        image.save(output)
        os.utime(output, None) 
        status = {"status": "success", "output_path": output, "prompt": prompt, "description": ''}
        if os.environ['BATCH'] == 'False':
            analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
            status['description'] = analysis['analysis']
        return status

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        gc.collect()
        if torch.cuda and torch.cuda.is_available():  # ✅ Was `if torch.cuda:` (always truthy)
            torch.cuda.empty_cache()

ImageGen = ImageGenQwen21

def GenerateImage(prompt='', output='tmp.png', width=WIDTH, height=HEIGHT, seed=SEED, imagegen=None):
    #prompt = EnhancePrompt('',prompt,'system/QwenImage.txt')['analysis']
    gen = imagegen if imagegen else ImageGen()
    if seed == -1:
        seed = random.randint(0, 1000000)
    prompt = expand_prompt_with_qwen_image(prompt)['rewritten_prompt']
    status = gen.generate(prompt, output, int(width), int(height), int(seed))
    del gen
    status['description'] = ''
    if os.environ['BATCH'] == 'False':
        analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
        status['description'] = analysis['analysis']
    status['prompt'] = prompt
    prompt_metadata(output, prompt)
    return status

ImageEdit = ImageEditQwen21

def EditImage(prompt='', images=[''], output='tmp_edit.png', width=WIDTH, height=HEIGHT, seed=SEED, img_edit=None):
    if not img_edit:
        edit = ImageEdit()
    else:
        edit = img_edit
    status = edit.generate(prompt, images, output, int(width), int(height), int(seed))
    if not img_edit:
        del edit
    return status

def shot_prompt(shot_type: str, character_count: int) -> str:
    shot_type = shot_type.lower()

    if shot_type == "closeup":
        return (
            "close-up shot, character face dominates frame, "
            "cinematic framing, clear subject focus"
        )

    if shot_type == "medium":
        return (
            "medium shot, waist-up framing, "
            "subject centered naturally within environment"
        )

    if shot_type == "wide":
        return (
            "wide shot, environment clearly visible, "
            "characters naturally integrated into the location"
        )

    if shot_type == "two_shot":
        return (
            "medium two-shot, both characters clearly visible, "
            "balanced composition, natural conversational spacing"
        )

    if shot_type == "ots" and character_count == 2:
        return (
            "over-the-shoulder shot, foreground character <image 2> partially visible, back of head and top of shoulders only "
            "focus on the other character <image 3>"
        )
    if shot_type == "ots" and character_count == 1:
            return (
            "over-the-shoulder shot, foreground character <image 2> partially visible, back of head and top of shoulders only "
            "focus is a perspective view of the character <image 2> is currently seeing"
        )

    return shot_type.replace("_", " ")


def build_qwen_scene_prompt(
    shot_type: str,
    action: str,
    character_count: int,
    style: str = "realistic",
) -> str:

    lines = []

    lines.append(
        "Use image 1 as the location reference."
    )

    if character_count > 0:
        lines.append(
            f"Use images 2 through {character_count + 1} as character identity references."
        )

    lines.append("")

    lines.append("SHOT:")
    lines.append(shot_prompt(shot_type, character_count))

    lines.append("")

    lines.append("ACTION:")
    lines.append(action)

    lines.append("")

    lines.append("COMPOSITION:")

    if shot_type == "two_shot":
        lines.extend([
            "Both characters visible.",
            "Characters facing each other naturally.",
            "Clear separation between subjects.",
            "Environment remains visible."
        ])

    elif shot_type == "ots":
        lines.extend([
            "Foreground character nearest camera.",
            "Background character clearly visible.",
            "Natural conversational framing."
        ])

    elif character_count == 1:
        lines.extend([
            "Single clear subject.",
            "Strong visual focus on the character.",
            "Natural integration into location."
        ])

    lines.append("")
    lines.append("STYLE:")
    lines.append(style)

    lines.append("")
    lines.append("Maintain character identity.")
    lines.append("Natural pose and body language.")
    lines.append("Consistent scene lighting.")
    lines.append("Photograph-like realism.")

    return "\n".join(lines)


def CompositeScene(
    background_path: str,
    characters: list[str],
    shot_type: str = "medium",
    action: str = "standing naturally",
    output: str = "composite.png",
    width: int = WIDTH,
    height: int = HEIGHT,
    seed: int = -1
):
    from pathlib import Path

    if not os.path.exists(background_path):
        raise FileNotFoundError(
            f"Background not found: {background_path}"
        )

    for char in characters:
        if not os.path.exists(char):
            raise FileNotFoundError(
                f"Character not found: {char}"
            )

    style = (
        "anime"
        if os.environ.get("ANIME", "False") != "False"
        else "realistic"
    )

    images = [background_path] + characters

    prompt = build_qwen_scene_prompt(
        shot_type=shot_type,
        action=action,
        character_count=len(characters),
        style=style,
    )

    print("\n=== SCENE PROMPT ===\n")
    print(prompt)

    expanded = expand_edit_prompt_with_qwen_image(image_paths=images,
        user_prompt=prompt
    )

    final_prompt = expanded["rewritten_prompt"]

    editor = ImageEditQwen21()

    status = editor.generate(
        prompt=final_prompt,
        images=images,
        output=str(output),
        width=width,
        height=height,
        seed=seed,
    )

    status["prompt"] = final_prompt

    return status

def add_metadata_char(imgpath, prompt='', seed=-1, generation_prompt=None):
    # Ensure the path is normalized for Linux/Windows safety
    cleaned_path = os.path.normpath(imgpath)

    from plan10.lib.util import wait_for_file
    wait_for_file(cleaned_path)

    # 1. READ STEP: Open, copy metadata, and immediately close/release the file handle
    with Image.open(cleaned_path) as target_image:
        metadata = load_metadata(target_image)
        img_copy = target_image.copy()

    # Build your prompt instructions
    base_instructions = '''
        Analyze the subject and describe ONLY clearly visible, literal traits. Return a single comma-separated string in this exact order: 
        subject_type, age_stage, ethnicity_origin, gender, skin_surface, face_shape, jawline, cheekbones, eyes, eyebrows, nose, lips,
        hair_fur_length_color_texture, hair_style, hairline, facial_hair_features, head_accessories, neck_accessories, eyewear, clothing,
        footwear, distinctive_markers.
        
        Rules:
        - Be exhaustive and hyper-accurate. Do NOT guess. If a trait isn't visible, use 'hidden_from_view'. If it doesn't apply, use 'not_applicable'.
        ...
        Respond ONLY with the string.
        '''

    if generation_prompt:
        base_instructions += f"""
        
        ADDITIONAL CONTEXT: 
        This character was generated using the following prompt. Use this prompt to identify the specific colors, materials, and distinctive features that were intentionally designed, even if they are subtle in the image:
        "{generation_prompt}"
        
        Ensure your description heavily aligns with the specific traits mentioned in this generation prompt.
        """

    # 2. ANALYSIS STEP: Send to your vision model analyzer
    analysis = AnalyzeImage(cleaned_path, base_instructions)
    raw = analysis['analysis'].strip().strip('"').strip("'")
    
    # Clean & filter without regex
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    cleaned = [p for p in parts if p.lower() not in ["none", "no glasses"]]
    clean_string = ", ".join(cleaned)
    
    # Update the metadata keys
    metadata.add_text("Description", clean_string)
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Seed", str(seed))
    if generation_prompt:
        metadata.add_text("GenerationPrompt", generation_prompt)
        
    # 3. WRITE STEP: Open a completely fresh file pointer strictly for saving
    img_copy.save(cleaned_path, pnginfo=metadata)
        
    return clean_string

    # Add this helper near your expander function
def prepare_image_for_expander(image_path: str, max_size: int = 512) -> Image.Image:
    """Resize image for the 7B expander - it only needs to understand content, not see every pixel."""
    img = Image.open(image_path).convert("RGB")
    
    # Resize to fit within max_size while maintaining aspect ratio
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    return img

def CreatePortrait(prompt='', reference='', output='character_tmp.png',
                    seed=-1, imagegen=None):
    """
    Generates a portrait. If reference provided, uses I2I expander + edit.
    If no reference, uses T2I expander + generation.
    """
    gen = imagegen if imagegen else ImageGen()
    
    width, height = (1024, 1024)
    
    if reference:
        # I2I path: analyze reference + edit
        print(f"[Portrait] Using I2I expander with reference...")
        user_prompt = f"Professional headshot portrait. Shoulders and head fully visible, front-facing, looking directly at camera. Clean neutral background, soft even studio lighting. {prompt}"
        
        try:
            i2i_result = expand_edit_prompt_with_qwen_image(
                image_paths=[reference],
                user_prompt=user_prompt
            )
            full_prompt = i2i_result['rewritten_prompt']
            print(f"[Portrait] Enhanced prompt ({len(full_prompt)} chars)")
            
            # Use EditImage with reference
            status = EditImage(
                prompt=full_prompt,
                images=[reference],
                output=str('tmp.png'),
                width=width,
                height=height,
                seed=seed
            )
        except Exception as e:
            print(f"[Portrait] I2I expander failed: {e}")
            print(f"[Portrait] Falling back to manual analysis...")
            full_prompt = _manual_portrait_analysis(reference, prompt)
            
            status = GenerateImage(
                prompt=full_prompt,
                output=str('tmp.png'),
                width=width,
                height=height,
                seed=seed,
                imagegen=gen
            )
    else:
        # T2I path: no reference, pure generation
        print(f"[Portrait] Using T2I expander (no reference)...")
        user_prompt = f"Professional headshot portrait. Shoulders and head fully visible, front-facing, looking directly at camera. Clean neutral background, soft even studio lighting. {prompt}"
        
        t2i_result = expand_prompt_with_qwen_image(user_prompt=user_prompt)
        full_prompt = t2i_result['rewritten_prompt']
        print(f"[Portrait] Enhanced prompt ({len(full_prompt)} chars)")
        
        status = GenerateImage(
            prompt=full_prompt,
            output=str('tmp.png'),
            width=width,
            height=height,
            seed=seed,
            imagegen=gen
        )
    
    # Remove background and save
    input_img = Image.open('tmp.png')
    output_tmp = remove(input_img)
    output_tmp.save(output)
    
    # Clean up
    if os.path.exists('tmp.png'):
        os.remove('tmp.png')
    
    status['prompt'] = full_prompt
    return status


def _manual_portrait_analysis(reference, prompt):
    """Fallback: manual analysis + prompt construction (original approach)."""
    analysis_prompt = (
        "Describe in a single, detailed sentence ONLY the character's "
        "facial features, hair style, hair length, and any identity-defining "
        "details (such as scars, freckles, makeup, or distinctive expressions). "
        "Ignore background, clothing below the shoulders, props, and lighting."
    )
    result = AnalyzeImage(reference, analysis_prompt)
    analysis_text = result.get('analysis', '').strip()
    if analysis_text:
        analysis_text = analysis_text[0].lower() + analysis_text[1:]
    
    full_prompt = (
        "Professional character reference headshot. Shoulders and head fully visible. "
        "Hair completely visible. Front-facing, looking directly at camera. "
        "Symmetrical face, highly detailed facial features, sharp focus on the face. "
        "Clean solid neutral background, soft even studio lighting. "
    )
    
    if analysis_text:
        full_prompt += (
            "The character's face and hair must match the following description: "
            f"{analysis_text}. "
        )
    
    if prompt:
        full_prompt += prompt
    
    return full_prompt

def CreateBackground(prompt='', output='location_tmp.png', seed=-1, override=None, ambience=True):
    seed = int(seed)
    
    classification = crowd_density(prompt)
    print(classification)
    print("CREATE BACKGROUND")
    
    if os.environ.get('FORCE_OVERRIDE', 'False') != 'False':
        override = (WIDTH, HEIGHT)
    
    user_part = prompt.strip() if prompt else "empty atmospheric location"
    
    # Let the expander handle EVERYTHING - location + ambience
    if ambience:
        expander_prompt = (
            f"Wide-angle establishing shot of {user_part}. "
            f"Include ambient background figures with {classification['density']} crowd density. "
            f"Figures should suggest {classification['activity']} activity. "
            "No focal subject, no characters in foreground, panoramic environmental view."
        )
    else:
        expander_prompt = (
            f"Wide-angle establishing shot of {user_part}. "
            "Completely empty, no people, no silhouettes, unoccupied space. "
            "No focal subject, panoramic environmental view."
        )
    
    print(f"[Background] Enhancing full background description...")
    t2i_result = expand_prompt_with_qwen_image(user_prompt=expander_prompt)
    
    final_prompt = t2i_result['rewritten_prompt']
    print(f"[Background] Enhanced prompt ({len(final_prompt)} chars)")
    print(final_prompt)
    
    gen = ImageGen()
    width, height = override if override else (1344, 768)
    
    status = gen.generate(final_prompt, output, width, height, seed)
    del gen
    
    status['description'] = add_metadata_loc(output, final_prompt, seed)
    status['prompt'] = prompt
    return status

def CreateCharacterSheet(prompt='', output='character_tmp.png', seed=-1, imagegen=None, override=None):
    seed = int(seed)
    width, height = override if override else (1536, 1536)
    user_prompt = (
        "Professional character design turnaround sheet, single image with two side-by-side views "
        "(3/4 front view and back view) of the same character. "
        "The character is in a neutral standing position, full body with a neutral expression. "
        "The character is standing on a seamless white cyclorama studio backdrop with soft volumetric "
        "studio lighting from above, creating subtle soft ground shadows beneath the character. "
        "Ensure the clothing, garment structure, proportions, and details match exactly between "
        "the front and back views. Clean crisp edges, professional concept art quality, "
        "sharp focus, high detail. "
        f"Character description: {prompt}" )
    
    # Single T2I expander call - it understands "turnaround sheet" natively
    #print(f"[Character Sheet] Enhancing prompt...")
    #t2i_result = expand_prompt_with_qwen_image(
    #    user_prompt = user_prompt
    #)
    
    #enhanced_prompt = t2i_result['rewritten_prompt']
    #print(f"[Character Sheet] Enhanced prompt ({len(enhanced_prompt)} chars)")
    
    # Generate directly
    gen = imagegen if imagegen else ImageGen()
    status = gen.generate(user_prompt, 'tmp.png', width, height, seed)
    if not imagegen:
        del gen
    
    # Remove background and save
    input_img = Image.open('tmp.png')
    output_tmp = remove(input_img)
    output_tmp.save(str(output))
    
    # Clean up
    if os.path.exists('tmp.png'):
        os.remove('tmp.png')
    
    status['description'] = add_metadata_char(output, prompt, seed)
    status['prompt'] = prompt
    
    return status

def sanitize_json(text):
    text = text.strip()
    # Remove backticks if present
    if text.startswith("```"):
        text = text.strip("`")
    # Remove accidental prose before/after JSON
    start = text.find("{")
    end = text.rfind("}") + 1
    return text[start:end]


def crowd_density(prompt=""):
    question = f'''
You are a classification engine.

Your job has two steps:

1. Identify the environment type being described.
   Return a single noun such as "nightclub", "warehouse", "forest",
   "gym", "museum", "street", "office", "hangar", etc.
   Ignore architectural emptiness, camera framing, and layout details.

2. Infer the typical human population density and typical background
   activity for that environment based on real-world knowledge.
   Do NOT use the prompt's emptiness or layout to suppress population.
   The ambience flag will handle suppression separately.

Return ONLY this JSON:

{{
  "environment": "<environment_type>",
  "density": "none" | "low" | "medium" | "high",
  "activity": "<inferred typical background activity>"
}}

Environment description:
"{prompt}"

'''

    answer = llm_analyze_media('',question)['analysis']
    print(answer)
    return json.loads(sanitize_json(answer))

def add_metadata_loc(imgpath, prompt='', seed=-1, brief=False, update=True):
    cleaned_path = os.path.normpath(imgpath)

    from plan10.lib.util import wait_for_file
    wait_for_file(cleaned_path)
    
    # 1. READ STEP: Open, collect info/metadata, and close immediately
    with Image.open(cleaned_path) as target_image:
        img_memory = target_image.copy()
        metadata = load_metadata(target_image)
        # Safely grab 'Brief' while the handle is open
        existing_brief = target_image.info.get('Brief', '')
        

    analysis_prompt = '''
Extract a structured spatial description of this BACKGROUND image.
CRITICAL: This image contains NO PEOPLE. Describe ONLY the environment.

Return ONLY the following fields:

1. VIEW_GEOMETRY — view height, angle, lens feel, depth cues.
2. GLOBAL_LAYOUT — foreground/midground/background partitioning and major planes.
3. ANCHOR_OBJECTS — fixed, non-movable environmental elements with positions (furniture, fixtures, architecture).
4. MATERIAL_CUES — environmental surfaces, textures, architectural materials (wood, stone, metal, fabric on furniture). DO NOT describe clothing or people.
5. LIGHTING_MODEL — direction, softness, color temperature, shadow behavior.
6. ATMOSPHERE — weather, haze, particulate, ambient mood.
7. COLOR_PROFILE — dominant palette and contrast profile.

Keep each field to 1 concise sentence. ABSOLUTELY NO CHARACTERS, NO PEOPLE, NO CLOTHING DESCRIPTIONS.
'''

    if brief:
        # Check the variable we safely extracted earlier
        if not existing_brief or update:
            bg_brief = AnalyzeImage(cleaned_path, "Description, Style, lighting, weather in <15 words.")['analysis'].strip()
            metadata.add_text("Brief", bg_brief)
            
            if update:
                # 2A. WRITE STEP (Brief path): Use a fresh file handler to save
                img_memory.save(cleaned_path, pnginfo=metadata).save(cleaned_path, pnginfo=metadata)
            return bg_brief
        else:
            return existing_brief

    # 3. ANALYSIS STEP (Full path)
    bg_analysis = AnalyzeImage(cleaned_path, analysis_prompt)
    bg_desc = bg_analysis['analysis'].strip()
    
    if update:
        metadata.add_text("Description", bg_desc)
        metadata.add_text("Prompt", prompt)
        metadata.add_text("Seed", str(seed))
        

        img_memory.save(cleaned_path, pnginfo=metadata).save(cleaned_path, pnginfo=metadata)
            
    return bg_desc

def main():
    import argparse, os
    os.environ['BATCH'] = 'True'
    parser = argparse.ArgumentParser()
    parser.add_argument('-W', '--width', type=int, default=WIDTH, help='width of output')
    parser.add_argument('-H', '--height', type=int, default=HEIGHT, help='height of output')
    parser.add_argument('-E', '--seed', type=int, default=SEED, help='seed')
    parser.add_argument('--portrait', action='store_true')
    parser.add_argument('-P', '--prompt', type=str, default='a beautiful woman tanning at the beach', help='prompt')
    parser.add_argument('-O', '--output', type=str, default='output.png')
    parser.add_argument('-C', '--character-sheet', action='store_true')
    parser.add_argument('-L', '--location', action='store_true')
    parser.add_argument('-I', '--images', action='append', default=[])
    parser.add_argument('-T', '--edit', action='store_true')
    parser.add_argument('-B', '--background', type=str, help='Background path')
    parser.add_argument('--chars', action='append', default=[], help='Character paths (1-2)')
    parser.add_argument('-S', '--shot-type', type=str, default='medium_single')
    parser.add_argument('-A', '--action', type=str, help='Action to complete')
    parser.add_argument('--expand-image', action='store_true')
    parser.add_argument('--composite', action='store_true')
    args = parser.parse_args()
    if args.character_sheet:
        print(CreateCharacterSheet(args.prompt, args.output, args.seed))
    elif args.location:
        print(CreateBackground(args.prompt, args.output,args.seed))
    elif args.expand_image:
        if args.edit:
            expanded = expand_edit_prompt_with_qwen_image(args.images, args.prompt)
            EditImage(expanded['rewritten_prompt'], args.images, args.output, args.seed)
        else:
            print(expand_prompt_with_qwen_image(args.prompt))
    elif args.edit:
        print(EditImage(args.prompt, args.images, args.output, args.width, args.height, args.seed))
    elif args.portrait:
        print(CreatePortrait('', args.images[0], args.output, args.seed))
    elif args.composite:
        print(CompositeScene(args.background, args.chars, args.shot_type, args.action, args.output, args.width, args.height, args.seed))
    else:
        print(GenerateImage(args.prompt, args.output, args.width, args.height, args.seed))

if __name__ == '__main__':
    main()
