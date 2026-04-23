from qwen_llm import llm_analyze_media

def AnalyzeImageSchema():
    return  {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze an image and return a text description.",
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

def AnalyzeImage(image='', prompt='Describe this.', output=None):
    status = llm_analyze_media(image, prompt)
    if output:
        from pathlib import Path
        Path(output).write_text(status['analysis'])
    return status

def EnhancePrompt(image='', prompt='a beautiful woman', enhancer='', output=None):
    from pathlib import Path
    eprompt = Path(enhancer).read_text()
    status = llm_analyze_media(image, prompt, eprompt)
    if output:
        Path(output).write_text(status['analysis'])
    return status

if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser(description='Fix Images.')
    parser.add_argument('-I', '--image', type=str, default='', help='Image to analyze')
    parser.add_argument('-P', '--prompt', type=str, default='Describe this.', help='prompt')
    parser.add_argument('-E', '--enhance', type=str, default=None, help='prompt enhancer')
    parser.add_argument('-O', '--output', type=str, default=None, help='file to output')
    args = parser.parse_args()
    if args.enhance:
        print(EnhancePrompt(args.image, args.prompt, args.enhance, args.output))
    else:
        print(AnalyzeImage(args.image, args.prompt))
