from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import gc
import torch
import os
from image_analysis import AnalyzeImage, EnhancePrompt
from config import load_environ

load_environ()

model = "black-forest-labs/FLUX.2-klein-4B"

class ImageGen(object):
    def __init__(self,vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
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
            vram_limit=vrlimit,
        )

    def generate(self, prompt, output, width, height, seed):
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


    def __del__(self):
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()

def GenerateImage(prompt='', output='tmp.png', width=1328, height=1328, seed=42):
    prompt = EnhancePrompt('',prompt,'system/QwenImage.txt')
    gen = ImageGen()
    status = gen.generate(prompt['analysis'], output, int(width), int(height), int(seed))
    del gen
    status['description'] = ''
    if os.environ['BATCH'] == 'False':
        analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
        status['description'] = analysis['analysis']
    status['prompt'] = prompt['analysis']
    return status

def CreateCharacterSheet(prompt='', output='character_tmp.png',seed=-1):
    seed=int(seed)
    prompt = (
    "create a character sheet single image with two side by side views "
    "(3/4 front view, back view) with plain white background, studio lighting. "
    "Ensure the clothing, legwear, sock length, and garment structure match exactly "
    "between the front and back views. "
    f"of {prompt}")
    gen = ImageGen()
    status = gen.generate(prompt, output, 1328, 1328, seed)
    del gen
    analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
    status['description'] = analysis['analysis']
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
    analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
    status['description'] = analysis['analysis']
    status['prompt'] = final_prompt
    return status


def GenerateReverseBackground(source_image: str, output: str = "reverse_bg.png", width: int = 1328, height: int = 1328, seed: int = -1):
    if not os.path.exists(source_image): 
        raise FileNotFoundError(f"Source not found: {source_image}")
    
    # Analysis: explicitly separate Anchors, Landmarks, and Lighting
    analysis_prompt = (
        "Describe scene in 4 parts:\n"
        "1. ENV: Indoor/Outdoor, lighting quality/direction, time/weather, style.\n"
        "2. SOURCES: Sun/lamp visible in frame? Position?\n"
        "3. LANDMARKS: Distant/prominent background features visible? Description.\n"
        "4. SPATIAL: Nearby structural anchors (railings, walls, pillars) with positions: LEFT/RIGHT/CENTER.\n"
        "Keep under 120 words total."
    )
    analysis = AnalyzeImage(source_image, analysis_prompt)
    env_desc = analysis['analysis'].strip()
    
    # Prompt: Landmarks vanish, Anchors flip, Lighting stays
    prompt = (
        f"{env_desc}\n\n"
        "Generate the reverse shot for a conversation scene: camera rotated 180° around the conversation axis.\n"
        "PRESERVE EXACTLY: environment type, lighting quality/direction, time of day, weather, materials, color palette.\n"
        "ADJUST SOURCES: Visible sun/lamps are now behind the camera and must NOT appear; keep only their lighting effect.\n"
        "HANDLE LANDMARKS: Major landmarks visible in the original frame are now behind the camera and must NOT appear. Replace with a background view consistent with the location but facing the opposite direction.\n"
        "INVERT SPATIAL: Nearby structural anchors (railings, walls, pillars) swap horizontal position—what was on the LEFT is now on the RIGHT, and vice versa.\n"
        "Cinematic, atmospheric, empty of characters or text."
    )
    
    return GenerateImage(prompt=prompt, output=output, width=width, height=height, seed=seed)

# ─────────────────────────────────────────────────────────────
# REVERSE BACKGROUND (Uses T2I, not Edit)
# ─────────────────────────────────────────────────────────────
def GenerateReverseBackgroundSchema():
    return {
        "type": "function", "function": {
            "name": "generate_reverse_background",
            "description": "Analyze a background and generate a NEW background from a different angle using text-to-image.",
            "parameters": {
                "type": "object", "properties": {
                    "source_image": {"type": "string", "description": "Path to source background to analyze."},
                    "output": {"type": "string", "default": "reverse_bg.png"},
                    "width": {"type": "integer", "default": 1280},
                    "height": {"type": "integer", "default": 720},
                    "seed": {"type": "integer", "default": -1}
                }, "required": ["source_image"]
            }
        }
    }



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
    parser = argparse.ArgumentParser(
                    prog='GenerateImages',
                    description='Generate images from prompt',
                    epilog='')
    parser.add_argument('-W', '--width', type=int, default=1024, help='width of output')
    parser.add_argument('-H', '--height', type=int, default=1024, help='height of output')
    parser.add_argument('-E', '--seed', type=int, default=42, help='seed')
    parser.add_argument('-P', '--prompt', type=str, default='a beautiful woman tanning at the beach', help='prompt')
    parser.add_argument('-O', '--output', type=str, default='output.png')
    parser.add_argument('-C', '--charactersheet', action='store_true')
    parser.add_argument('-L', '--location', action='store_true')
    parser.add_argument('-R', '--gen-reverse', action='store_true', help='Generate reverse-angle background (T2I)')
    args = parser.parse_args()
    if args.gen_reverse:
        if not args.images: print("ERROR: -I required for reverse gen"); exit(1)
        print(GenerateReverseBackground(args.images[0], args.output, args.width, args.height, args.seed))
    elif args.charactersheet:
        print(CreateCharacterSheet(args.prompt, args.output, args.seed))
    elif args.location:
        print(CreateBackground(args.prompt, args.output,args.seed))
    else:
        print(GenerateImage(args.prompt, args.output, args.width, args.height, args.seed))
