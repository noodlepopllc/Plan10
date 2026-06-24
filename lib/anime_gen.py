from diffsynth.pipelines.krea2 import Krea2Pipeline, ModelConfig
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig
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

class ImageGenKrea2(object):
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
            self.pipe = Krea2Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="krea/Krea-2-Turbo", origin_file_pattern="turbo.safetensors", **vram_config),
                    ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors", **vram_config),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
                ],
                tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
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
                num_inference_steps=8, 
                cfg_scale=1, 
                mu=1.15
            )
        image.save(output)
        return {"status":"success", "output_path":output}

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        gc.collect()
        if torch.cuda.is_available():  # ✅ Was `if torch.cuda:` (always truthy)
            torch.cuda.empty_cache()

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

class ImageGenAnima(object):
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
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }
            self.pipe = AnimaImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/diffusion_models/anima-base-v1.0.safetensors", **vram_config),
                    ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/text_encoders/qwen_3_06b_base.safetensors", **vram_config),
                    ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/vae/qwen_image_vae.safetensors", **vram_config),
                ],
                tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-0.6B", origin_file_pattern="./"),
                tokenizer_t5xxl_config=ModelConfig(model_id="stabilityai/stable-diffusion-3.5-large", origin_file_pattern="tokenizer_3/"),
                vram_limit=self.vrlimit,
            )


    def generate(self, prompt, output, width, height, seed):
        if not self.pipe:
            self.__enter__()
        image = self.pipe(
                prompt=prompt,
                seed=seed,
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



ImageGen = ImageGenKrea2

def GenerateImage(prompt='', output='tmp.png', width=WIDTH, height=HEIGHT, seed=42):
    gen = ImageGen()
    style_prefix = (
        "anime illustration, cel-shaded, flat colors, limited palette, "
        "clean lineart, manga-inspired, studio lighting, "
        "no photorealism, no 3D render, no realistic shading, "
    )
    status = gen.generate(
        f'{style_prefix}{prompt}',
        output, int(width), int(height), int(seed)
    )
    del gen
    status['description'] = ''
    if os.environ['BATCH'] == 'False':
        analysis = AnalyzeImage(
            output,
            "Briefly describe this anime image in under 100 words."
        )
        status['description'] = analysis['analysis']
    status['prompt'] = prompt
    return status

def add_metadata_char(imgpath, prompt='', seed=-1):
    target_image = Image.open(imgpath)
    metadata = PngInfo()

    combined_prompt = (
        '''
        You are analyzing an ANIME / MANGA style illustration.
        Describe ONLY clearly visible traits. Return a single comma-separated string
        in this exact order:

        subject_type, age_stage, gender_presentation, anime_substyle,
        face_shape, eye_shape, eye_color, eye_details, eyebrow_style,
        nose_style, mouth_style, expression, skin_tone, skin_finish,
        hair_length, hair_color, hair_style, bangs_style, hair_accessories,
        head_accessories, eyewear, body_type, clothing, footwear, signature_props.

        Rules:
        - Be accurate to what is VISIBLE in the anime art. Do NOT infer realistic anatomy.
        - If a trait is not visible or does not apply, write 'neutral'.
        - Output ONLY the comma-separated string, nothing else.

        === FIELD DEFINITIONS ===

        subject_type:
            anime_character, anime_anthropomorphic (furry/kemonomimi),
            anime_mecha_pilot, chibi, anime_robot, masked_anime, neutral

        age_stage:
            child, young_teen, teen, young_adult, adult, loli_archetype,
            shota_archetype, ageless, neutral

        gender_presentation:
            male, female, androgynous, ambiguous, neutral

        anime_substyle:
            shoujo, shounen, seinen, josei, chibi, 90s_retro,
            modern_slice_of_life, fantasy_anime, mecha_anime,
            magical_girl, visual_novel, neutral

        face_shape:
            V_line (most common anime), oval, heart, round, angular,
            soft_round, sharp_chin, neutral

        eye_shape (anime eyes are typically LARGE — do NOT describe as "narrow" unless truly so):
            large_round, large_almond, tsurime (upturned), tareme (downturned),
            sharp, droopy, cat_eye, half_lidded, huge_shoujo, narrow, neutral

        eye_color:
            specific color(s) — e.g. emerald_green, crimson_red, sapphire_blue,
            amber, violet, heterochromia (specify colors), gradient_iris, neutral

        eye_details:
            detailed_iris, starry_highlights, glowing, ringed_pupil,
            cat_slit_pupil, spiral, empty_white, visor_eyes, neutral

        eyebrow_style:
            thin, thick, arched, straight, angular, minimal_absent
            (many anime characters have barely-visible brows), neutral

        nose_style (anime noses are OFTEN minimal — do NOT force "medium/large"):
            dot, small_line, minimal_shadow, button, small, medium,
            sharp, hidden, realistic (rare in anime), neutral

        mouth_style:
            small, medium, full, minimal_line, fang_visible, cat_mouth,
            open_smile, closed_smile, neutral

        expression:
            neutral, smiling, serious, angry, sad, embarrassed_blush,
            determined, playful, tsundere, kuudere, crying, surprised, neutral

        skin_tone:
            porcelain, fair, light, tan, olive, dark, pale,
            blue_tinted, green_tinted (non-human), neutral

        skin_finish:
            cel_shaded, soft_shaded, matte, glossy, blush_prominent,
            freckled, scarred, neutral

        hair_length:
            very_short, short, medium, shoulder_length, long, very_long,
            waist_length, floor_length, neutral

        hair_color:
            black, brown, blonde, silver, white, pink, blue, red, green,
            purple, orange, multitone, gradient, streaked, pastel,
            unnatural_color (specify), neutral

        hair_style (use anime-specific terms):
            twin_tails, ahoge (antenna), drill_curls, hime_cut, bob,
            long_straight, spiky, messy, high_ponytail, low_ponytail,
            side_braid, french_braid, bun, half_up, side_swept,
            undercut, bowl_cut, bedhead, windblown, flowing,
            cat_ears_hair, wolf_cut, neutral

        bangs_style:
            straight_across, side_parted, blunt, wispy, split,
            asymmetrical, swept, no_bangs, neutral

        hair_accessories:
            ribbon, hairpin, flower, bow, hair_ornament, hair_ties,
            beads, feathers, crown_tiara, hair_bell, neutral

        head_accessories:
            cat_ears, animal_ears, horns, halo, headband, hat, helmet,
            mask_partial, mask_full, headphones, halo, neutral

        eyewear:
            glasses, sunglasses, monocle, goggles, visor, eyepatch,
            colored_contacts, none, neutral

        body_type (anime proportions, NOT realistic):
            slender, petite, athletic, muscular, curvy, chibi_proportions,
            tall, average, loli_body, bishounen, neutral

        clothing:
            describe simply — school_uniform, sailor_uniform, maid_outfit,
            military_uniform, fantasy_armor, magical_girl_outfit,
            kimono, yukata, hoodie, tshirt, dress, cloak, tactical_gear,
            futuristic_suit, neutral

        footwear:
            loafers, sneakers, boots, heels, sandals, barefoot,
            thigh_high_boots, geta, neutral

        signature_props:
            any iconic visible item — katana, staff, book, wand,
            plush_toy, weapon, musical_instrument, neutral

        === CRITICAL ANIME RULES ===
        1. Anime eyes are LARGE by default. Only use "narrow" if they are genuinely
           drawn narrow (e.g. a stern seinen character). Never describe typical
           anime eyes as "small" or "narrow".
        2. Anime noses are OFTEN a dot, line, or shadow. Do NOT describe as
           "medium" or "large" unless clearly drawn that way.
        3. Hair in anime is GRAVITY-DEFYING and CHUNKY. Use anime-specific
           terms (ahoge, twin_tails, drill_curls) instead of realistic ones.
        4. Skin in anime is CEL-SHADED with prominent blush. Prefer
           "cel_shaded" over realistic skin descriptors.
        5. For kemonomimi (animal-ear humans): use anime_anthropomorphic,
           describe animal ears under head_accessories, keep human face traits.
        6. For full furry/anthro anime: use anime_anthropomorphic, map
           muzzle->mouth_style, fur->hair, paws->footwear.
        7. NEVER force realistic defaults. If hair color is clearly pink,
           write "pink", not "dyed blonde". If eyes are glowing red, write
           "crimson_red" + "glowing".
        8. Chibi characters: use chibi for subject_type AND chibi_proportions
           for body_type.

        === EXAMPLES ===

        Shoujo heroine:
        "anime_character, young_teen, female, shoujo, V_line, huge_shoujo,
        emerald_green, detailed_iris starry_highlights, thin, dot, small,
        embarrassed_blush, fair, cel_shaded blush_prominent, very_long,
        blonde, long_straight, side_parted, ribbon, none, none, slender,
        school_uniform, loafers, neutral"

        Shounen protagonist:
        "anime_character, teen, male, shounen, sharp_chin, sharp,
        crimson_red, glowing, thick, small_line, medium, determined,
        tan, cel_shaded, short, black, spiky, split, none, none, none,
        athletic, orange_jacket, sneakers, katana"

        Magical girl:
        "anime_character, young_teen, female, magical_girl, V_line,
        large_round, sapphire_blue, starry_highlights, thin, dot, small,
        playful, porcelain, cel_shaded blush_prominent, very_long, pink,
        twin_tails drill_curls, wispy, bow, tiara, none, petite,
        magical_girl_outfit, thigh_high_boots, wand"

        Kemonomimi (cat-girl):
        "anime_anthropomorphic, teen, female, modern_slice_of_life,
        V_line, tsurime, amber, cat_slit_pupil, thin, dot, cat_mouth,
        playful, fair, cel_shaded blush_prominent, long, silver, messy,
        side_parted, hair_bell, cat_ears, none, slender, oversized_hoodie,
        barefoot, plush_toy"

        Mecha pilot:
        "anime_character, young_adult, male, mecha_anime, angular,
        sharp, ice_blue, ringed_pupil, thick, small, medium, serious,
        light, cel_shaded, short, silver, undercut, swept, none, helmet,
        none, athletic, plugsuit, magnetic_boots, neutral"

        Chibi:
        "chibi, child, female, chibi, round, large_round, brown,
        simple, minimal_absent, dot, cat_mouth, smiling, fair,
        cel_shaded blush_prominent, medium, brown, bob, blunt,
        flower, none, none, chibi_proportions, yellow_sundress,
        red_shoes, neutral"

        Respond ONLY with the comma-separated string.
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

def add_metadata_loc(imgpath, prompt='', seed=-1):
    target_image = Image.open(imgpath)
    metadata = PngInfo()

    loc_prompt = (
        "You are analyzing an ANIME background illustration (no characters present). "
        "Describe it in a single comma-separated string in this exact order, "
        "keeping it under 25 words total: "
        "environment_type, time_of_day, weather_atmosphere, lighting_style, "
        "color_palette, anime_substyle, mood. "
        "Use anime-native terms: e.g. 'shinkai_sky', 'ghibli_painterly', "
        "'golden_hour', 'volumetric_light', 'cherry_blossoms', "
        "'cyberpunk_neon', 'cel_shaded', 'watercolor'. "
        "Respond ONLY with the comma-separated string."
    )

    bg_analysis = AnalyzeImage(imgpath, loc_prompt)
    bg_desc = bg_analysis['analysis'].strip().strip('"').strip("'")

    metadata.add_text("Description", bg_desc)
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Seed", str(seed))
    target_image.save(imgpath, pnginfo=metadata)
    return bg_desc

def CreateCharacterSheet(prompt='', output='character_tmp.png', seed=-1, imagegen=None):
    seed = int(seed)
    prompt = (
        "anime character reference sheet, two side-by-side views "
        "(3/4 front view, back view), plain white background, "
        "clean lineart, cel-shaded flat colors, consistent design "
        "between front and back. "
        f"Character: {prompt}"
    )
    gen = imagegen if imagegen else ImageGen()
    status = gen.generate(prompt, output, 2048, 2048, seed)
    if not imagegen:
        del gen
    status['description'] = add_metadata_char(output, prompt, seed)
    status['prompt'] = prompt
    return status

# Optional style presets — pass via prompt prefix or as a separate arg
ANIME_BG_STYLES = {
    "shinkai": (
        "Makoto Shinkai style background, hyper-detailed sky, volumetric god rays, "
        "lens flare, light particles floating in air, dramatic cloud formations, "
        "golden hour or twilight color grading, cinematic depth of field, "
        "photorealistic lighting with anime color palette"
    ),
    "ghibli": (
        "Studio Ghibli style background, painterly watercolor textures, "
        "lush natural scenery, soft diffused sunlight, warm pastoral atmosphere, "
        "hand-painted look, gentle brushstrokes visible, nostalgic countryside"
    ),
    "slice_of_life": (
        "anime slice-of-life background, detailed everyday environment, "
        "soft natural lighting, gentle shadows, lived-in atmosphere, "
        "warm color palette, quiet contemplative mood"
    ),
    "fantasy": (
        "anime fantasy background, magical atmosphere, floating light particles, "
        "ethereal glow, mystical environment, vibrant saturated colors, "
        "dramatic sky, otherworldly scenery"
    ),
    "cyberpunk": (
        "anime cyberpunk background, neon-lit cityscape, rain-slicked streets, "
        "holographic signs, moody blue and magenta lighting, "
        "futuristic urban environment, atmospheric fog"
    ),
    "post_apoc": (
        "anime post-apocalyptic background, overgrown ruins, dramatic sky, "
        "melancholic atmosphere, nature reclaiming civilization, "
        "golden sunset through broken structures"
    ),
    "night": (
        "anime night scene background, starry sky, moonlit atmosphere, "
        "soft blue-purple color grading, city lights or lanterns in distance, "
        "quiet nocturnal mood"
    ),
    "default": (
        "anime background art, cel-shaded with painterly touches, "
        "rich atmospheric lighting, detailed environment, "
        "cinematic composition, vibrant but harmonious color palette"
    ),
}


def CreateBackground(
    prompt='',
    output='location_tmp.png',
    seed=-1,
):
    style = os.environ.get('STYLE','default')
    time_of_day=None       # optional: 'dawn', 'morning', 'noon', 'golden_hour', 'sunset', 'twilight', 'night'
    weather=None           # optional: 'clear', 'cloudy', 'rain', 'snow', 'fog', 'cherry_blossoms', 'autumn_leaves'
    seed = int(seed)
    print("CREATE BACKGROUND")

    # 1. Style anchor — the single most important line
    style_anchor = ANIME_BG_STYLES.get(style, ANIME_BG_STYLES["default"])

    # 2. Time-of-day modifier
    tod_map = {
        "dawn":        "early dawn, pink and lavender sky, first light",
        "morning":     "bright morning light, clear sky, fresh atmosphere",
        "noon":        "high noon, bright overhead sun, sharp shadows",
        "golden_hour": "golden hour, warm amber sunlight, long soft shadows",
        "sunset":      "sunset, orange and crimson sky, silhouetted horizon",
        "twilight":    "twilight, deep blue and purple sky, first stars",
        "night":       "night time, moonlit, deep blue atmosphere",
    }
    tod_line = f", {tod_map[time_of_day]}" if time_of_day in tod_map else ""

    # 3. Weather / atmosphere modifier
    weather_map = {
        "clear":            "clear sky",
        "cloudy":           "overcast sky, soft diffused light",
        "rain":             "falling rain, wet surfaces, raindrops, puddle reflections",
        "snow":             "falling snow, snow-covered surfaces, cold breath atmosphere",
        "fog":              "thick atmospheric fog, mist, low visibility, mysterious",
        "cherry_blossoms":  "falling cherry blossom petals, sakura, spring atmosphere",
        "autumn_leaves":    "falling autumn leaves, orange and red foliage, fall atmosphere",
    }
    weather_line = f", {weather_map[weather]}" if weather in weather_map else ""

    # 4. User subject
    user_part = prompt.strip() if prompt else "atmospheric empty location"

    # 5. Hard NO-CHARACTERS enforcement — tight and effective
    no_chars = (
        "ABSOLUTELY NO characters, people, humans, figures, silhouettes, "
        "animals, creatures, faces, or living beings anywhere in the frame. "
        "Pure empty scenery only."
    )

    # 6. Assemble final prompt
    final_prompt = (
        f"{style_anchor}{tod_line}{weather_line}. "
        f"Scene: {user_part}. "
        f"{no_chars} "
        f"High quality anime background art, detailed environment painting."
    ).strip()

    gen = ImageGen()
    status = gen.generate(final_prompt, output, 1920, 1080, seed)
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
