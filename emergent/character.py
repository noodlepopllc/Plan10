from lib.image_analysis import AnalyzeImage

class CharacterProfile:
    def __init__(self, character_ref_path):
        self.ref_path = character_ref_path
        self.characters = self._extract_all_characters()
    
    def _extract_all_characters(self):
        prompt = """Analyze this image and extract a complete profile for the 1 to 3 MOST PROMINENT FOREGROUND characters ONLY.

CRITICAL RULES:
1. IGNORE background people, crowds, blurry figures, or distant subjects.
2. Focus ONLY on characters who are large, in focus, and clearly the main subjects of the image.
3. Maximum of 3 characters. If there is only 1 prominent person, output only CHARACTER_1.

For EACH prominent character, provide:
1. VISUAL_ID: 15-25 word description including ethnicity, exact age range, hair color and style (length, texture), skin tone, face shape, distinctive facial features, and main clothing items with specific colors
2. APPEARANCE: Physical details (ethnicity, age, gender, face shape, eye shape, hair, body type, distinctive features)
3. CLOTHING: Detailed clothing (top color/style/fit, bottom color/style/fit, shoes, accessories, hair details)

Output format (use this EXACT structure):
CHARACTER_1:
VISUAL_ID: [15-25 word detailed description]
APPEARANCE: [description]
CLOTHING: [description]

CHARACTER_2: (ONLY if a second prominent foreground character exists)
VISUAL_ID: [15-25 word detailed description]
APPEARANCE: [description]
CLOTHING: [description]

CHARACTER_3: (ONLY if a third prominent foreground character exists)
VISUAL_ID: [15-25 word detailed description]
APPEARANCE: [description]
CLOTHING: [description]

Be extremely specific about colors, styles, and physical features. This will be used to maintain consistency across camera angles."""
        
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
                if current_char:
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
            elif current_field and current_char:
                current_char[current_field] += ' ' + line
        
        if current_char and current_char.get('visual_id'):
            characters.append(current_char)
        
        return characters
        
    def get_character(self, index):
        if 0 <= index < len(self.characters):
            return self.characters[index]
        return None
        
    def get_visual_id(self, index=0):
        char = self.get_character(index)
        return char['visual_id'] if char else "unknown character"