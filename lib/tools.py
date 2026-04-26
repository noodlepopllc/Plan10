import sys, torch, gc, random, time, os
from pathlib import Path
sys.path.append('./lib')
from image_gen import (
    CreateCharacterSheetSchema, 
    CreateBackgroundSchema, 
    CreateCharacterSheet, 
    CreateBackground, 
    GenerateImage, 
    GenerateImageSchema, 
    GenerateReverseBackgroundSchema, 
    GenerateReverseBackground
) 
from image_to_video import GenerateVideoSchema, GenerateVideo
from speech_to_video import GenerateTalkingVideo, GenerateTalkingVideoSchema
from image_analysis import EnhancePrompt
from dialog import VoiceDesignSchema, VoiceCloneSchema, DesignVoice, CloneVoice
from compositor import CompositeSceneSchema, CompositeScene
import traceback

# UPDATED IMPORTS to match consolidated image_edit.py
from image_edit import (
    EditImageSchema, 
    EditImage
)

# =============================================================================
# PARAMETER VALIDATION & FIXES
# =============================================================================

VALID_VOICE_ITEMS = {
    'child', 'teenager', 'young adult', 'middle-aged', 'elderly',
    'male', 'female',
    'very low pitch', 'low pitch', 'moderate pitch', 'high pitch', 'very high pitch', 'whisper',
    'american accent', 'british accent', 'australian accent', 'canadian accent',
    'chinese accent', 'indian accent', 'japanese accent', 'korean accent',
    'portuguese accent', 'russian accent'
}

def fix_voice_parameter(voice):
    """Auto-correct invalid voice parameters."""
    items = [x.strip() for x in voice.split(',')]
    valid_items = [x for x in items if x.lower() in VALID_VOICE_ITEMS]
    if not valid_items:
        return 'female, young adult, moderate pitch, british accent'
    return ', '.join(valid_items)


class ToolHandler(object):
    # UPDATED TOOLS LIST
    TOOLS = [
        CreateCharacterSheetSchema(),
        CreateBackgroundSchema(),
        GenerateImageSchema(),
        CompositeSceneSchema(),
        GenerateReverseBackgroundSchema(),
        EditImageSchema(), 
        GenerateVideoSchema(),
        GenerateTalkingVideoSchema(),
        VoiceDesignSchema(),
        VoiceCloneSchema()
    ]

    def detect_asset_type(self, path, tool_name):
        ext = Path(path).suffix.lower()
        if ext in {'.txt', '.md', '.json'}: return 'text'
        if ext in {'.png', '.jpg', '.jpeg', '.webp'}: return 'image'
        if ext in {'.mp4', '.mov', '.avi', '.webm'}: return 'video'
        if ext in {'.wav', '.mp3'}: return 'audio'
        if 'analyze' in tool_name: return 'text'
        if 'edit' in tool_name or 'generate' in tool_name: return 'image'
        if 'video' in tool_name: return 'video'
        if 'audio' in tool_name: return 'audio'
        return 'unknown'

    def register_asset(self, ctx, path, tool, args, result, explicit_alias=None):
        if not path or not os.path.exists(path):
            return None
        ctx.setdefault("assets", {})
        alias = explicit_alias or f"asset_{len(ctx['assets']) + 1}"
        asset_type = self.detect_asset_type(path, tool)
        
        ctx["assets"][alias] = {
            "path": path, 
            "type": asset_type,
            "description": result.get('description', ''),
            "metadata": {
                "tool": tool,
                "seed": args.get("seed"),
                "prompt": args.get("prompt", "")
            }
        }
        print(f"📦 Registered: {alias} [{asset_type}] -> {path}")
        return alias

    def resolve_asset(self, ref, ctx):
        if not ref: return None
        ref = ref.strip('"').strip("'")
        if os.path.exists(ref): return ref
        
        assets = ctx.get("assets", {})
        if ref in assets:
            p = assets[ref].get("path")
            return p if p and os.path.exists(p) else None
        else: 
            alt = f'outputs/{ref}'
            return alt if os.path.exists(alt) else None
        return None

    def render_assets(self, ctx):
        assets = ctx.get("assets", {})
        if not assets: return "  (none)"
        lines = []
        for alias, info in assets.items():
            base = os.path.basename(info.get('path', 'unknown'))
            t = info.get('type', '?').upper()
            description = info.get('description', '')
            lines.append(f"  • {alias} [{t}], description: '{description}' -> {base}")
        return "\n".join(lines)

    def _handle_success(self, tool_name, filtered, chosen_alias, ctx, result, ext_override=None):
        os.makedirs("outputs", exist_ok=True)
        base = filtered.get("output", f"output_{int(time.time()*1000)}{ext_override or '.png'}")
        result["output_path"] = base
        result.setdefault("status", "success")
        new_alias = self.register_asset(ctx, base, tool_name, filtered, result, explicit_alias=chosen_alias)
        return {
            "status": "success",
            "asset_alias": new_alias,
            "output_path": base,
            "description": result.get("description", ""),
            "prompt_used": result.get("prompt", filtered.get("prompt", ""))
        }

    def run_tool(self, tool_name, args, ctx):
        print(f"🔧 EXECUTING: {tool_name}")
        chosen_alias = args.pop('alias', None)
        
        # UPDATED VALID PARAMS
        VALID = {
            "create_character_sheet": ["prompt", "output", "alias"],
            "create_background": ["prompt", "output", "alias"],
            "generate_image": ["prompt", "width", "height", "seed", "output", "alias"],
            "composite_scene": ["background_path", "characters", "shot_type", "action", "width", "height", "output", "alias","seed"],
            "generate_reverse_background": ["source_image", "width", "height", "seed", "output", "alias"],
            "edit_image": ["images", "prompt", "width", "height", "seed", "output", "alias"],
            "image_to_video": ["prompt", "media", "width", "height", "seed", "duration_sec", "output", "alias"],
            "dialog_to_video": ["prompt", "media", "audio", "width", "height", "seed", "output", "alias"],
            "design_voice": ["text", "voice", "output", "duration", "seed"],
            "clone_voice": ["text", "audio", "output", "duration", "seed"],
        }
        
        filtered = {k: v for k, v in args.items() if k in VALID.get(tool_name, [])}
        if 'voice' in filtered:
            filtered['voice'] = fix_voice_parameter(filtered['voice'])
            
        # UPDATED ASSET RESOLUTION KEYS
        for key in ['images', 'characters', 'source_image', 'background_path', 'media', 'audio']:
            if key in filtered:
                if isinstance(filtered[key], list):
                    resolved = [self.resolve_asset(x, ctx) for x in filtered[key]]
                else:
                    resolved = self.resolve_asset(filtered[key], ctx)
                if not resolved:
                    return {"status": "error", "message": f"Could not resolve asset: {filtered[key]}"}
                filtered[key] = resolved

        try:
            os.makedirs("outputs", exist_ok=True)
            ts = int(time.time() * 1000)
            
            if tool_name == "create_character_sheet":
                filtered['output'] = f"outputs/char_{chosen_alias or ''}_{ts}.png"
                result = CreateCharacterSheet(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result)
                
            elif tool_name == "create_background":
                filtered['output'] = f"outputs/bg_{chosen_alias or ''}_{ts}.png"
                result = CreateBackground(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result)

            elif tool_name == "generate_image":
                filtered['output'] = f"outputs/gen_{chosen_alias or ''}_{ts}.png"
                result = GenerateImage(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result)
                
            elif tool_name == "composite_scene":
                filtered['output'] = f"outputs/comp_{chosen_alias or ''}_{ts}.png"
                result = CompositeScene(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result)
                
            elif tool_name == "generate_reverse_background":
                filtered['output'] = f"outputs/reverse_{chosen_alias or ''}_{ts}.png"
                result = GenerateReverseBackground(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result)

            elif tool_name == "edit_image":
                filtered['output'] = f"outputs/edit_{chosen_alias or ''}_{ts}.png"
                result = EditImage(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result)
                
            elif tool_name == "image_to_video":
                filtered['output'] = f"outputs/i2v_{chosen_alias or ''}_{ts}.mp4"
                result = GenerateVideo(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result, ext_override=".mp4")
                
            elif tool_name == "dialog_to_video":
                filtered['output'] = f"outputs/s2v_{chosen_alias or ''}_{ts}.mp4"
                result = GenerateTalkingVideo(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result, ext_override=".mp4")

            elif tool_name == "clone_voice":
                filtered['output'] = f"outputs/clonevoice_{chosen_alias or ''}_{ts}.wav"
                result = CloneVoice(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result, ext_override=".wav")

            elif tool_name == "design_voice":
                filtered['output'] = f"outputs/designvoice_{chosen_alias or ''}_{ts}.wav"
                result = DesignVoice(**filtered)
                return self._handle_success(tool_name, filtered, chosen_alias, ctx, result, ext_override=".wav")

            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
            
        except Exception as e:
            print(traceback.format_exc())
            return {"status": "error", "message": str(e)}
        finally:
            gc.collect()
            torch.cuda.empty_cache()