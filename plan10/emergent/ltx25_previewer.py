from plan10.lib.config import load_config
load_config()

from plan10.lib.image_analysis import AnalyzeImage
import re, os, math

BG_PROMPT = """
Describe only what is visible in the image in 80–120 words.

Use concise bullet points (6–8 bullets, 10–15 words each).

Include:
- terrain
- lighting
- palette
- atmosphere
- major objects

Then provide:
- one sentence describing the inferred visual aesthetic / art style

Do not exceed 120 words before the aesthetic line.
"""

CHAR_PROMPT = '''
Return ONE sentence in this exact format:

"The {race/ethnicity} {gender} with {hair style} {hair color} hair is wearing {clothing list} and {accessory list}."

Use ONLY these slots. Do not reorder them.

DEFINITIONS:
- {clothing list} includes the material of each item.
  Examples: "red satin shirt", "blue denim jeans", "black leather jacket", "tan canvas shorts".
- {accessory list} includes material when relevant.
  Examples: "silver metal collar", "brown leather satchel", "chrome visor".

'''

BRIEF_PROMPT = '''
Return ONE sentence in this exact format:

"The {hair color}-haired {gender} in a {clothing color} {clothing item}."

Use ONLY these slots. Do not reorder them.
'''


class LTXPipeline:
    def __init__(self, backend='smol', temperature=0.4, max_tokens=2048):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.run_length = 10.0

    # -----------------------------
    # Parsing
    # -----------------------------
    def parse_beat_assets(self, text: str):
        assets = []
        for line in text.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue

            asset_type = parts[0]
            asset_id   = parts[1]
            asset_path = parts[2]
            description = parts[3]

            assets.append({
                "type": asset_type,
                "id": asset_id,
                "path": asset_path,
                "description": description
            })
        return assets

    def extract_shots(self, text: str):
        shots = []
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.lower().startswith("shot"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue

            description = parts[1]
            try:
                duration = float(parts[2])
            except ValueError:
                duration = 0.0

            shots.append({
                "description": description,
                "duration": duration,
            })
        return shots

    # -----------------------------
    # Image Analysis
    # -----------------------------
    def analyze_background(self, path: str):
        result = AnalyzeImage(
            path,
            prompt="Describe the background, in detail include any structures",
            backend=self.backend
        )
        return result['analysis'].replace('\n', ' ')

    def analyze_character(self, path: str):
        result = AnalyzeImage(
            path,
            prompt=CHAR_PROMPT,
            backend=self.backend,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return result['analysis'].replace('\n', ' ')

    def analyze_brief(self, path: str):
        result = AnalyzeImage(
            path,
            prompt=BRIEF_PROMPT,
            backend=self.backend,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return result['analysis'].replace('\n', ' ')

    # -----------------------------
    # Formatting
    # -----------------------------
    def format_shots(self, shots, briefs):
        timestamp = 0.0
        shot_lines = []
        for i, shot in enumerate(shots):
            duration = float(shot['duration'])
            end_timestamp = timestamp + duration
            description = shot['description']

            for k, v in briefs.items():
                description = description.replace(k, f'{k} ({v})')

            shot_lines.append(
                f'{timestamp:05.2f}-{end_timestamp:05.2f} ( Shot {i+1} ) : {description}'
            )
            timestamp = end_timestamp
        self.run_length = math.ceil(end_timestamp)

        return '\n'.join(shot_lines)

    def build_ltx_prompt(self, style, bg_desc, char_descs, shots):
        characters = []
        for k, v in char_descs.items():
            characters.append(f'{k} - {v}')

        return f'''
ART STYLE/THEME:
{style}

ENVIRONMENT:
{bg_desc}
        
CHARACTERS:
{'\n\n'.join(characters)}

SHOTS:
{shots}
        '''

    # -----------------------------
    # Main Execution
    # -----------------------------
    def run(self, beat_path: str, style: str, use_descriptions=False):
        from pathlib import Path

        raw_beats = Path(beat_path).read_text()
        beatdata = self.parse_beat_assets(raw_beats)

        char_descs = {}
        briefs = {}
        bg_desc = ''

        for asset in beatdata:
            if asset['type'] == 'bg':
                if not use_descriptions:
                    bg_desc = self.analyze_background(asset['path'])
                else:
                    bg_desc = asset['description']
            elif asset['type'] == 'char':
                if not use_descriptions:
                    char_descs[asset['id']] = self.analyze_character(asset['path'])
                else:
                    char_descs[asset['id']] = asset['description']
                briefs[asset['id']] = self.analyze_brief(asset['path'])

        shots = self.extract_shots(raw_beats)
        formatted_shots = self.format_shots(shots, briefs)

        return self.build_ltx_prompt(style, bg_desc, char_descs, formatted_shots)

def main():
    import argparse, sys, json
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Convert beats to videos using LTX2.5')
    parser.add_argument('-B', '--beat', type=str, default='', help='Story beat to render')
    parser.add_argument('-O', '--output', type=str, default=None, help='file to output')
    parser.add_argument('-U', '--use-descriptions', action='store_true')
    parser.add_argument('-S', '--style', type=str, help='art style')
    args = parser.parse_args()
    output = args.output

    if not args.beat:
        print("You are a horrible person, beats are required. Shame, Shame on you.")
        sys.exit()
    if not output:
        output = args.beat.replace('.txt', '_ltx.txt')
    converter = LTXPipeline()
    converted = converter.run(args.beat, style=args.style, use_descriptions=args.use_descriptions)
    print(converted)
    Path(output).write_text(f'RUNLENGTH (s):{converter.run_length}\n{converted}')

if __name__ == '__main__':
    main()
