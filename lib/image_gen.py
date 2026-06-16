from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowMatchScheduler
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import gc
import torch
import os
from image_analysis import AnalyzeImage, EnhancePrompt
from config import load_environ
from PIL import Image
from PIL.PngImagePlugin import PngInfo

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))

class ImageGenZImage(object):
    def __init__(self,vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
        self.vrlimit = vrlimit
        self.pipe = None

    def __enter__(self):
        if not self.pipe:
            vram_config = {
                "offload_dtype": torch.bfloat16,
                "offload_device": "cpu",
                "onload_dtype": torch.bfloat16,
                "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }
            self.pipe = ZImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors", **vram_config),
                    ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
                    ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
                ],
                tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
                vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
            )


    def generate(self, prompt, output, width, height, seed):
        if not self.pipe:
            self.__enter__()
        image = self.pipe(
                prompt=prompt,
                seed=seed,
                cfg_scale=1.0,
                height=height,
                width=width
            )
        image.save(output)
        return {"status":"success", "output_path":output}

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        gc.collect()
        if torch.cuda.is_available():  # ✅ Was `if torch.cuda:` (always truthy)
            torch.cuda.empty_cache()

class ImageGenQwen(object):
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
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": "cpu",
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }
            self.pipe = QwenImagePipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(model_id="Qwen/Qwen-Image-2512", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
                        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
                        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
                ],
                    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
                    vram_limit=self.vrlimit,
                )
            self.pipe.load_lora(self.pipe.dit, "./loras/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors", alpha=1.0)
            self.pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")


    def generate(self, prompt, output, width, height, seed):
        if not self.pipe:
            self.__enter__()
        image = self.pipe(
                prompt=prompt,
                seed=seed,
                num_inference_steps=8,
                cfg_scale=1.0,
                height=height,
                width=width
            )
        image.save(output)
        return {"status":"success", "output_path":output}

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        gc.collect()
        if torch.cuda.is_available():  # ✅ Was `if torch.cuda:` (always truthy)
            torch.cuda.empty_cache()

class ImageGenKlein(object):
    def __init__(self,vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
        self.vrlimit = vrlimit
        self.pipe = None

    def __enter__(self):
        model = "black-forest-labs/FLUX.2-klein-4B"
        if not self.pipe:
            vram_config = {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": "cpu",
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }

            self.pipe = Flux2ImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id=model, origin_file_pattern="text_encoder/*.safetensors", **vram_config),
                    ModelConfig(model_id=model, origin_file_pattern="transformer/*.safetensors", **vram_config),
                    ModelConfig(model_id=model, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
                ],
                tokenizer_config=ModelConfig(model_id=model, origin_file_pattern="tokenizer/"),
                vram_limit=self.vrlimit,
            )
        return self

    def generate(self, prompt, output, width, height, seed):
        if not self.pipe:
            self.__enter__()

        image = self.pipe(
                prompt=prompt,
                seed=seed,
                num_inference_steps=4,
                cfg_scale=1.0,
                height=height,
                width=width
            )
        image.save(output)
        return {"status":"success", "output_path":output}

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        gc.collect()
        if torch.cuda and torch.cuda.is_available():  # ✅ Was `if torch.cuda:` (always truthy)
            torch.cuda.empty_cache()

if os.environ.get("IMAGE_GEN", "KLEIN") == "KLEIN":
    ImageGen = ImageGenKlein
elif os.environ.get("IMAGE_GEN", "ZIMAGE") == "ZIMAGE":
    ImageGen = ImageGenZImage
else:
    ImageGen = ImageGenQwen

def GenerateImage(prompt='', output='tmp.png', width=WIDTH, height=HEIGHT, seed=42):
    #prompt = EnhancePrompt('',prompt,'system/QwenImage.txt')['analysis']
    gen = ImageGen()
    status = gen.generate(prompt, output, int(width), int(height), int(seed))
    del gen
    status['description'] = ''
    if os.environ['BATCH'] == 'False':
        analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
        status['description'] = analysis['analysis']
    status['prompt'] = prompt
    return status

def add_metadata_char(imgpath, prompt='', seed=-1):
    target_image = Image.open(imgpath)
    metadata = PngInfo()

    combined_prompt = (
        '''
        Analyze the subject and describe ONLY clearly visible traits. Return a single comma-separated string in this exact order: 
        subject_type, age_stage, ethnicity_origin, skin_surface, face_shape, jawline, cheekbones, eyes, eyebrows, nose, lips, 
        hair_fur_length_color_texture, hair_style, hairline, facial_hair_features, head_accessories, eyewear, clothing, footwear.
        
        Rules:
        - Be accurate. Do NOT guess. If a trait isn't visible or doesn't apply, write 'neutral'.
        - subject_type: human, anthropomorphic, android, masked, heavily_stylized, neutral
        - age_stage: child, youth, young adult, adult, elderly, timeless, neutral
        - ethnicity_origin: east asian, south asian, middle eastern, african, european, latinx, fantasy_race, machine_origin, neutral
        - skin_surface: fair, light, medium, tan, deep, metallic, synthetic, fur, scales, painted, masked, neutral
        - face_shape: oval, round, heart, square, long, muzzle, angular, geometric, neutral
        - jawline: soft, defined, sharp, angular, mechanical, fur-lined, hidden, neutral
        - cheekbones: low, medium, high, structural, hidden, neutral
        - eyes: almond, round, narrow, wide-set, glowing, lens, visor, painted, hidden, neutral
        - eyebrows: straight, arched, thick, thin, painted, mechanical, fur, hidden, neutral
        - nose: small, medium, large, narrow, wide, snout, vent, painted, hidden, neutral
        - lips: thin, medium, full, painted, sealed, mechanical, hidden, neutral
        - hair_fur_length_color_texture: short/medium/long + color + straight/wavy/curly/coarse, OR fur: short/long + color + dense/patchy, OR synthetic: fiber/metallic + color, OR 'neutral'
        - hair_style: ponytail, bun, braid, tied-back, loose, half-up, bob, pixie, crew cut, buzz cut, fade, undercut, slicked back, messy, short crop, comb over, mane, tufted, helmet-integrated, none, neutral
        - hairline: straight, widow's peak, rounded, receding, fur-edge, seam-line, masked, neutral
        - facial_hair_features: clean-shaven, stubble, mustache, beard, goatee, sideburns, fur_muzzle, mechanical_grille, painted, hidden, neutral
        - head_accessories: ribbons, bandana, hats, helmet, mask_partial, mask_full, crown, none, neutral
        - eyewear: glasses, sunglasses, visor, goggles, none, neutral
        - clothing: describe visible items simply: yellow sundress, white tshirt, armored vest, etc.
        - footwear: white tennis shoes, red heels, mechanical boots, paw-pads, none, neutral

        Critical Rules:
        1. If subject_type is masked/heavily_stylized: prioritize describing what is VISIBLE through/around the mask or makeup.
        2. If subject_type is anthropomorphic: map human-equivalent terms (e.g., muzzle for nose, fur for hair, paw-pads for footwear).
        3. If subject_type is android: use mechanical/synthetic descriptors where applicable; 'neutral' for biological terms that don't apply.
        4. NEVER force human defaults: if hair isn't visible, write 'none' or 'neutral', NOT 'bob' or 'pixie'.
        5. For heavy makeup: describe the painted/applied appearance, not the underlying biology.

        Examples:
        Human female: "human, adult, european, light, oval, defined, high, almond, arched, medium, full, long brown wavy hair, low ponytail, straight, clean-shaven, black ribbons, none, navy uniform, black kitten heels"
        Human male: "human, young adult, east asian, light, square, sharp, medium, narrow, thick straight, medium, thin, short black straight hair, short crop fade, straight, clean-shaven, none, none, gray button-up, black shoes"
        Anthropomorphic wolf: "anthropomorphic, adult, fantasy_race, gray fur, muzzle, defined, structural, almond glowing, fur brows, snout, hidden, long gray dense fur, loose mane, fur-edge, fur_muzzle, none, none, leather harness, paw-pads"
        Android: "android, timeless, machine_origin, metallic silver, geometric, angular, structural, lens eyes, mechanical, vent, sealed, none, none, seam-line, mechanical_grille, helmet-integrated, visor, armored plating, magnetic boots"
        Masked hero: "masked, adult, neutral, masked, neutral, hidden, hidden, almond visible, neutral, hidden, painted, neutral, neutral, masked, hidden, mask_partial, none, none, tactical suit, combat boots"
        Heavy makeup drag: "heavily_stylized, adult, neutral, painted fair, heart, sharp, high, wide painted, arched thick, small contoured, full painted, long blonde straight wig, high ponytail, straight, painted, crown, dramatic lashes, sequined gown, platform heels"

        Respond ONLY with the string.
        '''
    )

    analysis = AnalyzeImage(imgpath, combined_prompt)
    raw = analysis['analysis'].strip().strip('"').strip("'")
    
    # Clean & filter without regex
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # Remove "none"/"no glasses" so diffusion doesn't accidentally render them
    cleaned = [p for p in parts if p.lower() not in ["none", "no glasses"]]
    clean_string = ", ".join(cleaned)
    metadata.add_text("Description", clean_string)
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Seed", str(seed))
    target_image.save(imgpath, pnginfo=metadata)
    return clean_string

def add_metadata_loc(imgpath, prompt='', seed=-1, brief=False):
    target_image = Image.open(imgpath)
    metadata = PngInfo()
    analysis_prompt = '''
Extract a structured spatial description of this background image.
Return ONLY the following fields:

1. CAMERA_GEOMETRY — camera height, angle, lens feel, depth cues.
2. GLOBAL_LAYOUT — foreground/midground/background partitioning and major planes.
3. ANCHOR_OBJECTS — fixed, non-movable environmental elements with positions.
4. MATERIAL_CUES — surfaces, textures, architectural materials.
5. LIGHTING_MODEL — direction, softness, color temperature, shadow behavior.
6. ATMOSPHERE — weather, haze, particulate, ambient mood.
7. COLOR_PROFILE — dominant palette and contrast profile.

Keep each field to 1 concise sentence. No characters, no narrative.
    '''
    if brief:
        bg_brief = AnalyzeImage(imgpath, "Description, Style, lighting, weather in <15 words.")['analysis'].strip()
        metadata.add_text("Brief", bg_brief)
        return bg_brief
    bg_analysis = AnalyzeImage(imgpath, analysis_prompt)
    bg_desc = bg_analysis['analysis'].strip()
    metadata.add_text("Description", bg_desc)
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Seed", str(seed))
    target_image.save(imgpath, pnginfo=metadata)
    return bg_desc

def CreateCharacterSheet(prompt='', output='character_tmp.png',seed=-1, imagegen=None):
    seed=int(seed)
    prompt = (
    "create a character sheet single image with two side by side views "
    "(3/4 front view, back view) with plain white background, studio lighting. "
    "Ensure the clothing and garment structure match exactly "
    "between the front and back views."
    f"of {prompt}")
    gen = imagegen if imagegen else ImageGen()
    status = gen.generate(prompt, output, 1328, 1328, seed)
    if not imagegen:
        del gen
    status['description'] = add_metadata_char(output, prompt, seed)
    status['prompt'] = prompt
    return status

def CreateBackground(prompt='', output='location_tmp.png',seed=-1):
    seed=int(seed)
    print("CREATE BACKGROUND")
    base_prompt = (
        "A pure environmental background plate with absolutely no characters, people, animals, or foreground subjects. "
        "Focus solely on scenery, lighting, atmosphere, and spatial composition. "
    )
    
    # 2. Combine user input + hard exclusion cues
    user_part = prompt.strip() if prompt else "empty, atmospheric location"
    combined = f"{base_prompt} {user_part}"
    
    exclusion_suffix = (
        " Composition: edge-to-edge environment, uniform depth, no central focal point, background-only layout. "
        " Exclude: figures, faces, silhouettes, text, logos, living beings, narrative subjects."
    )
    final_prompt = (combined + exclusion_suffix).strip() + "When enhancing background prompts, preserve all exclusion constraints. Do not add people, animals, or narrative subjects. Maintain environmental/empty scene focus."
    gen = ImageGen()
    status = gen.generate(final_prompt, output, 1328, 1328, seed)
    del gen
    status['description'] = add_metadata_loc(output, final_prompt, seed)
    status['prompt'] = final_prompt
    return status






def GenerateImageSchema():
    return {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate a new image from a text prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image description."},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "seed": {"type": "integer"}
                },
                "required": ["prompt"]
            }
        }
    }

def CreateCharacterSheetSchema():
    return {
        "type": "function",
        "function": {
            "name": "create_character_sheet",
            "description": "Generate a character reference sheet with side-by-side 3/4 front and back views on a white background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string", 
                        "description": "Detailed description of the character's appearance, clothing, and style.",
                        "seed": {"type": "integer"}
                    }
                },
                "required": ["prompt"]
            }
        }
    }

def CreateBackgroundSchema():
    return {
        "type": "function",
        "function": {
            "name": "create_background",
            "description": "Generate a pure environmental background plate with NO characters, subjects, or foreground objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string", 
                        "description": "Description of the environment, lighting, and atmosphere (e.g., 'cyberpunk city street at night, wet asphalt').",
                        "seed": {"type": "integer"}
                    }
                },
                "required": ["prompt"]
            }
        }
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-W', '--width', type=int, default=WIDTH, help='width of output')
    parser.add_argument('-H', '--height', type=int, default=HEIGHT, help='height of output')
    parser.add_argument('-E', '--seed', type=int, default=42, help='seed')
    parser.add_argument('-P', '--prompt', type=str, default='a beautiful woman tanning at the beach', help='prompt')
    parser.add_argument('-O', '--output', type=str, default='output.png')
    parser.add_argument('-C', '--character-sheet', action='store_true')
    parser.add_argument('-L', '--location', action='store_true')
    args = parser.parse_args()
    if args.character_sheet:
        print(CreateCharacterSheet(args.prompt, args.output, args.seed))
    elif args.location:
        print(CreateBackground(args.prompt, args.output,args.seed))
    else:
        print(GenerateImage(args.prompt, args.output, args.width, args.height, args.seed))
