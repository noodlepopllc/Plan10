#!/usr/bin/env python3
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_environ
load_environ()
from image_analysis import AnalyzeImage

def analyze_scene(image_path):
    prompt = """Analyze this scene and generate a story context based on what you see.

LOCATION/SETTING: Describe the environment, time of day, mood, atmosphere.
CHARACTER BODY LANGUAGE: For each visible character, describe their posture, facial expression, and physical relationship to others (e.g., leaning in, arms crossed, facing each other).
STORY CONTEXT: Based on the location and body language, write 2-3 sentences describing what is likely happening and the emotional dynamic. This should feel like a natural continuation of the visual setup.

Output ONLY the "STORY CONTEXT:" section. Do not output the location or body language sections. Keep it to 2-3 sentences."""

    try:
        result = AnalyzeImage(image_path, prompt)
        # Clean up the output to ensure it's just the context
        context = result['analysis'].strip()
        if context.upper().startswith("STORY CONTEXT:"):
            context = context.split(":", 1)[1].strip()
        return context
    except Exception as e:
        print(f"Error analyzing scene: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an image and extract story context.")
    parser.add_argument('-I', '--image', type=str, required=True, help="Path to the image to analyze")
    args = parser.parse_args()
    
    print(analyze_scene(args.image))