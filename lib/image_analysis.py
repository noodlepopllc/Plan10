from lib.qwen_llm import llm_analyze_media
import os, re
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
    final_audio_prompt = f"{cleaned_string}, close microphone perspective, non-verbal, purely physical sound effects, no speech, no music"
    
    return final_audio_prompt

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
    """Analyze image or video using SmolVLM2."""
    model, processor = load_smol_vlm()
    
    # Determine if it's a video or image
    media_path = str(media)
    ext = Path(media_path).suffix.lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Build message content
    if is_video:
        content = [
            {"type": "video", "path": media_path},
            {"type": "text", "text": prompt}
        ]
    else:
        content = [
            {"type": "image", "path": media_path},
            {"type": "text", "text": prompt}
        ]
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            max_new_tokens=max_tokens
        )
    
    # Decode only the generated portion
    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]
    
    # Cleanup
    del inputs, generated_ids
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

def AnalyzeImage(image='', prompt='Describe this.', output=None, backend=None):
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
    if backend is None:
        backend = os.environ.get("VISION_BACKEND", "qwen").lower()
    
    if backend == "smol":
        analysis_text = analyze_with_smol(image, prompt)
        status = {'analysis': analysis_text}
    else:
        # Default to Qwen
        status = llm_analyze_media(image, prompt)
    
    if output:
        Path(output).write_text(status['analysis'])
    
    return status

def EnhancePrompt(image='', prompt='a beautiful woman', enhancer='', output=None, backend=None):
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
    if not os.path.exists(enhancer):
        repo_root = Path(__file__).parent.parent
        enhancer = repo_root / "system" / Path(enhancer).name

    eprompt = Path(enhancer).read_text()
    
    if backend is None:
        backend = os.environ.get("VISION_BACKEND", "qwen").lower()
    
    if backend == "smol":
        analysis = analyze_with_smol(image, prompt, temperature=0.5)
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
    parser.add_argument('-B', '--backend', type=str, default=None, choices=['qwen', 'smol'], 
                       help='Vision backend to use (default: from VISION_BACKEND env or qwen)')
    args = parser.parse_args()
    
    if args.enhance:
        print(EnhancePrompt(args.image, args.prompt, args.enhance, args.output, args.backend))
    else:
        result = AnalyzeImage(args.image, args.prompt, output=args.output, backend=args.backend)
        print(result['analysis'])


if __name__ == '__main__':
    main()
