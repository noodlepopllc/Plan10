import sys
sys.path.append('./lib')

from qwen_llm import llm_generate_pipeline
from pathlib import Path

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-I', '--input', type=str, default='', help='input file')
    p.add_argument('-O', '--output', type=str, default='', help='output file')
    args = p.parse_args()

    story = Path(args.input).read_text()
    llm_generate_pipeline(story, output_path=args.output)
