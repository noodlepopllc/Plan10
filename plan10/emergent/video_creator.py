import sys, os
import argparse
from pathlib import Path

from plan10.lib.config import load_environ
load_environ()

from plan10.emergent.state_manager import StateManager
from plan10.emergent.character import CharacterProfile
from plan10.emergent.pipeline import Pipeline
from plan10.lib.scene_analyzer import analyze_scene
from plan10.lib.util import extract_frame, load_metadata
from PIL import Image

from plan10.lib.decomposer import decompose_scene

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))
ANIME = os.environ.get('ANIME', 'False') != 'False'
MMH3 = os.environ.get('MMH3', 'False') != 'False'
QWEN2 = os.environ.get('IMAGE_GEN', 'False') == 'QWEN2'


import os
from PIL import Image
from plan10.lib.util import load_metadata

def get_or_create_visual_id(character_image: str, goal: str):
    """Get cached visual ID or generate and cache it."""
    cleaned_path = os.path.normpath(character_image)
    
    # 1. READ STEP: Open the image and check for the cached ID
    with Image.open(cleaned_path) as img:
        img.load()
        visual_id = img.info.get("VisualID")
        
        # Pull out existing metadata structure right away
        metadata = load_metadata(img)
    
    # Initialize variables to prevent NameErrors down the line
    character_name = "Unknown" 

    # 2. GENERATION STEP: If not cached, run the profile processor
    if not visual_id:
        profile = CharacterProfile(cleaned_path, goal)
        visual_id = profile.get_visual_id(0)
        character_name = profile.get_character_name(0)
        
        # Update metadata object
        metadata.add_text("VisualID", visual_id)
        
        # 3. WRITE STEP: Open a clean, fresh file pointer exclusively to save
        with Image.open(cleaned_path) as out_img:
            out_img.save(cleaned_path, pnginfo=metadata)
    else:
        # If it was cached, we still need the character name. 
        # (Assuming you need to recreate the profile to fetch it if it isn't in metadata)
        profile = CharacterProfile(cleaned_path, goal)
        character_name = profile.get_character_name(0)

    return visual_id, character_name


if ANIME:
    from plan10.lib.anime_gen import GenerateImage, prompt_metadata
elif os.environ.get('IMAGE_GEN', 'KLEIN') == 'QWEN21':
    from plan10.lib.qwen21 import  GenerateImage, prompt_metadata
else:
    from plan10.lib.image_gen import GenerateImage, prompt_metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--ref', type=str, action='append', default=[])
    parser.add_argument('-p', '--portraits', type=str, action='append', default=[])
    parser.add_argument('-I', '--initial', type=str, default='')
    parser.add_argument('-P', '--prompt', type=str, default='')
    parser.add_argument('-C', '--context', type=str, default='')
    parser.add_argument('-O', '--output', type=str, default="feedback_output")
    parser.add_argument('-W', '--width', type=int, default=WIDTH)
    parser.add_argument('-H', '--height', type=int, default=HEIGHT)
    parser.add_argument('-S', '--seed', type=int, default=SEED)
    parser.add_argument('-M', '--scene-mode', action='store_true')
    parser.add_argument('-D', '--duration', type=int, default=5)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('-G', '--goal', type=str, default=None, 
        help='Narrative goal to work toward (e.g., "woman gets ready for work")')
    
    args, _ = parser.parse_known_args()
    state_mgr = StateManager(args.output)
    first_run = False
    
    # Load or Initialize State
    if state_mgr.exists() and not args.reset:
        print("🔄 Resuming from state file...")
        state = state_mgr.load()
        refs = state['character_refs']
        portraits = state['portraits']
        visual_ids = state['visual_ids']
        char_names = state['char_names']
        beat_count = state['beat_count']
        current_media = state['current_media']
        story_context = state['story_context']
        history = state['history']
        pending_setup = state['pending_setup']
        needs_transition = state['needs_transition']
        video_queue = state.get('video_queue', [])
        scene_mode = state.get('scene_mode', False) 
        initial = state['initial_media']
        current_bg = state['current_bg']
        goal = state.get('goal')
        duration = state.get('duration')
        output_dir = state.get('output_dir')
        
    else:
        scene_mode = args.scene_mode
        first_run = True
        goal = args.goal
            
        print("🆕 Starting new loop...")
        refs = args.ref
        portraits = args.portraits
        
        # Handle initial image generation if needed
        if not args.initial and not args.prompt:
            print("Error: --initial (-I) or --prompt (-P) required for a new run")
            sys.exit(-1)
            
        if not args.initial:
            if WIDTH > HEIGHT:
                GenerateImage(prompt=args.prompt, output=f'{args.output}/improv.png', width=1920, height=1088, seed=args.seed)
            else:
                GenerateImage(prompt=args.prompt, output=f'{args.output}/improv.png', width=1088, height=1920, seed=args.seed)
            initial = f'{args.output}/improv.png'
            current_media = initial
        else:
            _, image = extract_frame(args.initial, WIDTH, HEIGHT, 'last_frame.png')
            current_media = image
            initial = image
        current_bg = current_media

        if not refs:
            decompose_scene(
                input_image=current_media,
                prompt=prompt_metadata(current_media),
                output_dir=args.output,
                seed=args.seed
            )
            for p in ['character_1.png', 'character_2.png', 'character_3.png']:
                if Path(f'{args.output}/{p}').exists():
                    refs.append(f'{args.output}/{p}')
                    
        print(f"REFERENCES: {refs}")

        char_ids = [get_or_create_visual_id(ref, goal) for ref in refs]
        
        # Sanitize BOTH lists to ensure they stay perfectly in sync
        sanitized_char_ids = []
        char_names = []
        
        for i, (vid, name) in enumerate(char_ids, 1):
            # Clean the name: handle None, strip whitespace, make lowercase for comparison
            clean_name = (name or "").strip().lower()
            
            if not clean_name or clean_name == "unknown":
                final_name = f"char{i}"
            else:
                final_name = name  # Keep original valid name (preserves casing)
                
            sanitized_char_ids.append((vid, final_name))
            char_names.append(final_name)
            
        # Overwrite the originals with the sanitized versions
        char_ids = sanitized_char_ids
        visual_ids = [x[0] for x in char_ids]
        char_names = [x[1] for x in char_ids]

        tmp_names = []

        for i, name in enumerate(char_names, 1):
            if name.strip() == 'unknown':
                tmp_names.append(f'char{i}')
            else:
                tmp_names.append(name)
        char_names = tmp_names
        
        beat_count = 0
        story_context = args.context
        history = []
        pending_setup = None
        needs_transition = False
        video_queue = []

    # Check if there's a pending video that needs to be rendered first
    has_pending = any(job['status'] == 'pending' for job in video_queue)
    if has_pending:
        print("⏳ Waiting for video renderer to complete pending job...")
        print("   Run: python video_runner.py -O", args.output)
        sys.exit(0)  # Exit gracefully, don't proceed until video is ready

    context = story_context
    if not context:
        context = analyze_scene(current_media)

    output_dir = args.output
    
    # Initialize Pipeline
    pipeline = Pipeline(refs, args.output, args.width, args.height, args.seed, visual_ids, 
        scene_mode=scene_mode, goal=goal)
    pipeline.initial_media = initial
    if first_run and not args.scene_mode:
        from plan10.lib.util import resize_image
        background = f'{args.output}/background.png'
        if Path(background).exists():
            current_bg = background
        tmp = Image.open(current_media)
        tmp, _ = resize_image(tmp, max(WIDTH, HEIGHT), aspect_ratio=WIDTH/HEIGHT, return_pil=True)
        current_media = current_media.replace('.png','_resized.png')
        tmp.save(current_media)
    
    # Execute ONE creative step
    try:
        result = pipeline.execute_step(
            current_media, current_bg, context, beat_count, 
            history, pending_setup, needs_transition
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  Manual stop detected. Saving state...")
        state_mgr.save({
            "beat_count": beat_count, "current_media": current_media,
            "current_bg": current_bg, 
            "portraits": portraits,
            "story_context": context, "history": history,
            "pending_setup": pending_setup, "needs_transition": needs_transition,
            "character_refs": refs, "visual_ids": visual_ids, "char_names": char_names,
            "video_queue": video_queue,
            "scene_mode": scene_mode,  # <-- missing
            "output_dir": args.output, "width": args.width, "height": args.height, "seed": args.seed,
            "initial_media": str(initial),  # <-- missing
            "goal": goal,  # <-- missing
            "duration": args.duration  # <-- missing
        })
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(255)

        # Clean up old completed jobs (keep last 3 for reference)
    video_queue = [job for job in video_queue if job['status'] in ['pending', 'processing', 'bad render']] + \
                  [job for job in video_queue if job['status'] == 'complete'][-3:]

    if result['video_job']:
        # Add the new video job to the queue
        video_queue.append(result['video_job'])
    else:
        video_queue[-1]["status"] = "bad render"
        new_job = video_queue[-1].copy()
        new_job["status"] = "pending"
        new_job["beat"] += 1
        new_job["output_path"] = f"{output_dir}/beat_{new_job['beat']:03d}.mp4"
        video_queue.append(new_job)



    # Save state
    new_state = {
        "beat_count": result['beat_count'],
        "current_media": str(result['current_media']),
        "story_context": context,
        "history": result['history'][-3:],
        "pending_setup": result['pending_setup'],
        "needs_transition": result['needs_transition'],
        "character_refs": refs,
        "portraits": portraits,
        "visual_ids": visual_ids,
        "char_names": char_names,
        "video_queue": video_queue,
        "scene_mode": scene_mode,
        "output_dir": args.output,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "initial_media": str(initial),
        "current_bg": str(result['current_bg']), 
        "goal": goal,
        "duration": args.duration
    }
    state_mgr.save(new_state)
    
    print(f"✅ Beat {result['beat_count']} planned. Video queued for rendering.")
    sys.exit(result['beat_count'])

if __name__ == "__main__":
    main()