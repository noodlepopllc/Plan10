from plan10.lib.config import load_config
load_config()
from plan10.lib.qwen_llm import llm_analyze_media
import os, re, gc
import torch
from pathlib import Path

# Global model cache
_smol_model = None
_smol_processor = None

AUDIO_SYSTEM_PROMPT = """
You are a Sound Design Prompt Generator for an audio diffusion model. 
Take the visual description provided and translate it into a high-density, action-oriented sound effect prompt.

Rules:
1. Describe specific textures, physical movements, and acoustic perspective.
2. If human movement or clothing is mentioned, use highly specific foley verbs (e.g., "creaking", "scraping", "shuffling").
3. ABSOLUTELY NO SPEECH OR MUSIC: Use zero words that imply talking, dialogue, or musical elements.
4. Output format: A single line of comma-separated descriptions under 25 words.

Example Output: "Subtle candle wax crackling, heavy leather armor creaking with body movement, soft linen dress rustling, quiet stone room echo"
"""

AUDIO_SYSTEM_PROMPT = """
You are a Sound Design Prompt Generator for an audio diffusion model. 
Take the visual description provided and translate it into a high-density, action-oriented sound effect prompt.

Rules:
1. Describe specific textures, physical movements, and acoustic perspective.
2. If human movement or clothing is mentioned, use highly specific foley verbs (e.g., "creaking", "scraping", "shuffling").
3. ABSOLUTELY NO SPEECH OR MUSIC: Use zero words that imply talking, dialogue, or musical elements.
4. Output format: A single line of dominant background sound under 10 words.

Example Output: "background murmurs"
"""

def translate_to_audio_prompt(visual_prompt):
    if not visual_prompt: return ""
    
    # Get raw prompt from your Qwen wrapper
    raw_analysis = llm_analyze_media('', visual_prompt, AUDIO_SYSTEM_PROMPT)["analysis"]
    
    # Programmatic Hard Scrub (lowercase for perfect safety parsing)
    cleaned = raw_analysis.lower().strip()
    
    # Strip dangerous tokens that trigger Woosh vocal tracts
    banned_speech_words = r"\b(talking|speech|dialogue|dialog|whispering|murmuring|voice|voices|speaking|words)\b"
    cleaned = re.sub(banned_speech_words, "", cleaned)
    
    # Strip formatting junk (double commas, loose strings)
    items = [item.strip() for item in cleaned.split(",") if item.strip()]
    cleaned_string = ", ".join(items)
    
    # Force the strict negative constraints to the tail end of the string
    final_audio_prompt = f"{cleaned_string}, close microphone perspective, non-verbal, purely physical sound effects"
    
    return final_audio_prompt

import os
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

def AnalyzeMediaGemma(media='', prompt="Describe this", max_tokens=512, temperature=0.7):
    """
    Completely self-contained Gemma 4 backend function matching your unified signature.
    Leverages native automated file loading with full video audio-track routing.
    """
    GEMMA_PROCESSOR, GEMMA_MODEL = None, None
    
    # 1. Self-contained Lazy Initialization with Environment Profiling
    if GEMMA_MODEL is None or GEMMA_PROCESSOR is None:
        model_id = "google/gemma-4-12B-it"
        GEMMA_PROCESSOR = AutoProcessor.from_pretrained(model_id)
        
        vram_limit = int(os.environ.get("VRAM", 80))
        use_bnb = os.environ.get("BITSNBYTES", "False").strip().lower() in ["true", "1", "yes"]
        kwargs = {"device_map": "auto"}
        
        if use_bnb:
            if vram_limit < 16:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                kwargs["load_in_8bit"] = True
        else:
            kwargs["dtype"] = torch.bfloat16

        GEMMA_MODEL = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)

    # 2. Handle Text-Only Fallbacks
    if not media:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        load_audio = False
    else:
        media_src = str(Path(media).resolve()) if not (media.startswith("http://") or media.startswith("https://")) else media
        ext = Path(media_src).suffix.lower() if not media_src.startswith("http") else media_src
        is_video = any(e in ext for e in ['.mp4', '.avi', '.mov', '.mkv', '.webm'])
        load_audio = is_video # Only trigger audio extractor blocks if it's a video file

        # 3. Structural Asset Pipeline (Ordered: Video/Image Asset first, then Prompt Text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video" if is_video else "image", "video" if is_video else "image": media_src},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

    try:
        # 4. Tokenize and execute utilizing the critical load_audio_from_video parameter
        inputs = GEMMA_PROCESSOR.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            load_audio_from_video=load_audio,
        ).to(GEMMA_MODEL.device)
        
        input_len = inputs["input_ids"].shape[-1]
        do_sample = temperature > 0.0

        with torch.inference_mode():
            outputs = GEMMA_MODEL.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample
            )

        # 5. Extract text responses
        response = GEMMA_PROCESSOR.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    finally:
        # --- CRITICAL VRAM SWEEP HOOKS ---
        # Explicitly delete the heavy intermediate tensors holding activation maps
        if inputs is not None:
            del inputs
        if outputs is not None:
            del outputs
            
        # Clear out Python reference counters and clear the PyTorch allocator cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if GEMMA_MODEL is not None:
            # Move back to CPU before deletion to sever active CUDA streams
            GEMMA_MODEL.to("cpu")
        del GEMMA_MODEL
        GEMMA_MODEL = None
        if GEMMA_PROCESSOR is not None:
            del GEMMA_PROCESSOR
            GEMMA_PROCESSOR = None

    return response

def load_smol_vlm():
    """Load SmolVLM2 model and processor, caching them globally."""
    global _smol_model, _smol_processor
    
    if _smol_model is None:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        print(f"Loading SmolVLM2 model: {model_id}")
        
        _smol_processor = AutoProcessor.from_pretrained(model_id)
        _smol_model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        
        print(f"✓ SmolVLM2 loaded on cuda")
    
    return _smol_model, _smol_processor

def AnalyzeMedia(media='', prompt="Describe this", max_tokens=512, temperature=0.7):
    model, processor = load_smol_vlm()

    media_path = str(media)
    ext = Path(media_path).suffix.lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    if is_video:
        content = [{"type": "video", "path": media_path},
                   {"type": "text", "text": prompt}]
    elif not media:
        content = [{"type": "text", "text": prompt}]
    else:
        content = [{"type": "image", "path": media_path},
                   {"type": "text", "text": prompt}]

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            max_new_tokens=max_tokens,
            use_cache=True,   # <-- RESTORED
        )

    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    # Cleanup
    del inputs
    del generated_ids
    torch.cuda.empty_cache()

    return generated_text.strip()

def AnalyzeImageSchema():
    return  {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze an image or video and return a text description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Asset alias or file path to analyze."},
                    "prompt": {"type": "string", "description": "Question or focus for the analysis."}
                },
                "required": ["image", "prompt"]
            }
        }
    }

def AnalyzeImage(image='', prompt='Describe this.', output=None, backend=None, max_tokens=4096, temperature=0.5):
    """
    Analyze an image or video using either Qwen-VL or SmolVLM2.
    
    Args:
        image: Path to image or video file
        prompt: Analysis prompt
        output: Optional file path to save result
        backend: 'qwen' or 'smol'. If None, uses VISION_BACKEND env var (default: 'qwen')
    
    Returns:
        dict with 'analysis' key containing the text response
    """
    if not backend:
        backend = os.environ.get("VISION_BACKEND", "qwen").lower()
    
    if backend == "smol":
        analysis_text = AnalyzeMedia(image, prompt, max_tokens=max_tokens, temperature=temperature)
        status = {'analysis': analysis_text}
    elif backend == "gemma":
        analysis_text = AnalyzeMediaGemma(image, prompt, max_tokens=max_tokens, temperature=temperature)
        status = {'analysis': analysis_text}
    else:
        # Default to Qwen
        status = llm_analyze_media(image, prompt, max_tokens=max_tokens, temperature=temperature)
    
    if output:
        Path(output).write_text(status['analysis'])
    
    return status

def EnhancePrompt(image='', prompt='a beautiful woman', enhancer='', output=None, backend=None, ispath=True):
    """
    Enhance a prompt using image/video analysis.
    
    Args:
        image: Path to image or video file
        prompt: Base prompt to enhance
        enhancer: Path to file containing enhancement instructions
        output: Optional file path to save result
        backend: 'qwen' or 'smol'. If None, uses VISION_BACKEND env var
    
    Returns:
        Enhanced prompt text
    """
    if ispath:
        if not os.path.exists(enhancer):
            repo_root = Path(__file__).parent.parent
            enhancer = repo_root / "system" / Path(enhancer).name

        eprompt = Path(enhancer).read_text()
    else:
        eprompt = enhancer
    
    if backend is None:
        backend = os.environ.get("VISION_BACKEND", "qwen").lower()
    
    if backend == "smol":
        analysis = AnalyzeMedia(image, prompt, temperature=0.5)
        enhanced = f"{analysis}\n\nEnhancement instructions: {eprompt}"
        status = {'analysis': enhanced}
    else:
        status = llm_analyze_media(image, prompt, eprompt, temperature=0.5)
    
    if output:
        Path(output).write_text(status['analysis'])
    
    return status['analysis']

def main():
    import argparse, sys

    parser = argparse.ArgumentParser(description='Analyze Images or Videos.')
    parser.add_argument('-I', '--image', type=str, default='', help='Image or video to analyze')
    parser.add_argument('-P', '--prompt', type=str, default='Describe this.', help='prompt')
    parser.add_argument('-E', '--enhance', type=str, default=None, help='prompt enhancer')
    parser.add_argument('-O', '--output', type=str, default=None, help='file to output')
    parser.add_argument('-B', '--backend', type=str, default=None, choices=['qwen', 'smol', 'gemma'], 
        help='Vision backend to use (default: from VISION_BACKEND env or qwen)')
    parser.add_argument('-M', '--max-tokens', type=int, default=512)
    parser.add_argument('-T', '--temperature', type=float, default=0.5)
                       
    args = parser.parse_args()
    
    if args.enhance:
        print(EnhancePrompt(args.image, args.prompt, args.enhance, args.output, args.backend))
    else:
        result = AnalyzeImage(args.image, args.prompt, output=args.output, backend=args.backend, max_tokens=args.max_tokens, temperature=args.temperature)
        print(result['analysis'])

if __name__ == '__main__':
    main()
