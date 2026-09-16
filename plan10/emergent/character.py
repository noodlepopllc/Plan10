from plan10.lib.image_analysis import AnalyzeImage
from plan10.lib.qwen_llm import llm_analyze_media
import re, json

class CharacterProfile:
    def __init__(self, character_ref_path, seed_profile=None):
        self.ref_path = character_ref_path
        self.seed_profile = seed_profile or ""
        self.characters = self._extract_all_characters()
        self._match_seed_characters_llm()
    
    def _extract_all_characters(self):
        prompt = """Analyze this image and extract a complete profile for the 1 to 3 MOST PROMINENT FOREGROUND characters ONLY.

CRITICAL RULES:
1. IGNORE background people, crowds, blurry figures, or distant subjects.
2. Focus ONLY on characters who are large, in focus, and clearly the main subjects of the image.
3. Maximum of 3 characters. If there is only 1 prominent person, output only CHARACTER_1.

For EACH prominent character, provide:
1. VISUAL_ID: 15-25 word description including ethnicity, exact age range, hair color and style (length, texture), skin tone, face shape, distinctive facial features, and main clothing items with specific colors
2. APPEARANCE: Physical details
3. CLOTHING: Detailed clothing

Output format:
CHARACTER_1:
VISUAL_ID: ...
APPEARANCE: ...
CLOTHING: ...

CHARACTER_2:
...
"""
        
        result = AnalyzeImage(self.ref_path, prompt)['analysis'].strip()
        return self._parse_character_data(result)
    
    def _parse_character_data(self, text):
        characters = []
        current_char = {}
        current_field = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('CHARACTER_'):
                if current_char.get('visual_id'):
                    characters.append(current_char)
                current_char = {}
                current_field = None
            elif line.startswith('VISUAL_ID:'):
                current_field = 'visual_id'
                current_char['visual_id'] = line.split(':', 1)[1].strip()
            elif line.startswith('APPEARANCE:'):
                current_field = 'appearance'
                current_char['appearance'] = line.split(':', 1)[1].strip()
            elif line.startswith('CLOTHING:'):
                current_field = 'clothing'
                current_char['clothing'] = line.split(':', 1)[1].strip()
            elif current_field:
                current_char[current_field] += ' ' + line
        
        if current_char.get('visual_id'):
            characters.append(current_char)
        
        return characters


    def parse_seed_characters(self, seed_text):
        """
        Extracts character name + description pairs from the Characters: section
        of the seed profile.
        Returns a dict: { "Sora": "...", "Lindsy": "..." }
        """

        characters_section = []
        in_characters = False

        for line in seed_text.splitlines():
            line = line.strip()

            if line.lower().startswith("characters"):
                in_characters = True
                continue

            if in_characters:
                # Stop when we hit the next section
                if re.match(r"^(location|motivations|spark|story goal|initial situation)", line.lower()):
                    break
                characters_section.append(line)

        # Now parse lines like:
        # - Sora: description...
        # - Lindsy: description...
        char_dict = {}

        for line in characters_section:
            if line.startswith("- "):
                line = line[2:].strip()  # remove "- "
            if ":" in line:
                name, desc = line.split(":", 1)
                char_dict[name.strip()] = desc.strip()

        return char_dict


    def _match_seed_characters_llm(self):
        if not self.seed_profile:
            return
        
        seed_text = "\n".join(
            f"{name}: {desc}"
            for name, desc in self.parse_seed_characters(self.seed_profile).items()
        )

        for char in self.characters:
            vid = char.get("visual_id", "")
            
            # Base prompt setup
            base_prompt = (
                "You are a character‑matching assistant.\n\n"
                "You are given:\n"
                "1. A list of SEED CHARACTERS with names and detailed descriptions.\n"
                "2. A VISUAL_ID description extracted from an image.\n\n"
                "Your task:\n"
                "Determine which SEED CHARACTER the VISUAL_ID most closely matches.\n\n"
                "Return ONLY valid JSON in this exact format:\n"
                "{\n"
                '  "character_name": "<name>",\n'
                '  "confidence": <float between 0 and 1>\n'
                "}\n\n"
                f"SEED CHARACTERS:\n{seed_text}\n\n"
                f'VISUAL_ID:\n"{vid}"'
            )

            current_prompt = base_prompt
            max_retries = 2

            for attempt in range(max_retries):
                try:
                    # Execute API Call
                    response_data = llm_analyze_media(
                        '', 
                        prompt=current_prompt, 
                        max_tokens=1024, 
                        temperature=0.2
                    )
                    
                    raw_response = response_data.get('analysis', '').strip()
                    
                    # Safely isolate the JSON boundary blocks
                    start_idx = raw_response.find('{')
                    end_idx = raw_response.rfind('}')
                    
                    if start_idx == -1 or end_idx == -1:
                        raise ValueError("Missing JSON curly brace formatting structures.")
                        
                    clean_json_str = raw_response[start_idx:end_idx + 1]
                    parsed_json = json.loads(clean_json_str)
                    
                    # Apply parameters safely
                    char["character_name"] = parsed_json.get("character_name", "unknown")
                    
                    try:
                        char["confidence"] = float(parsed_json.get("confidence", 0.0))
                    except (ValueError, TypeError):
                        char["confidence"] = 0.0
                        
                    break  # Successful pass! Exit the retry loop
                    
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    if attempt < max_retries - 1:
                        # Safe concatenation string to prevent Python f-string bracket compilation crashes
                        truncated_error_payload = raw_response[:500] if raw_response else "Empty string payload"
                        
                        retry_prompt = (
                            "Return ONLY a valid JSON dictionary in this exact format:\n"
                            "{\n"
                            '  "character_name": "<name>",\n'
                            '  "confidence": <float>\n'
                            "}\n\n"
                            "CRITICAL CORRECTION TASK:\n"
                            f"Your last attempt failed parsing with error: \"{str(e)}\"\n"
                            "Do NOT repeat conversation logs or text history. Fix your syntax error immediately.\n\n"
                            "BROKEN TEXT PAYLOAD TO REWRITE:\n"
                            '"""\n' + truncated_error_payload + '\n"""\n'
                        )
                        current_prompt = retry_prompt
                    else:
                        # Hard failure fallback assignments
                        char["character_name"] = "unknown"
                        char["confidence"] = 0.0




    def get_character(self, index):
        if 0 <= index < len(self.characters):
            return self.characters[index]
        return None
        
    def get_visual_id(self, index=0):
        char = self.get_character(index)
        return char['visual_id'] if char else "unknown character"

    def get_character_name(self, index=0):
        char = self.get_character(index)
        return char.get("character_name", "unknown") if char else "unknown"
