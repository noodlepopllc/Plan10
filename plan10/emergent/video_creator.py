import sys, os
import argparse
from pathlib import Path

from plan10.lib.config import load_environ
load_environ()

from plan10.emergent.state_manager import StateManager
from plan10.emergent.character import CharacterProfile
from plan10.emergent.pipeline import Pipeline
from plan10.lib.scene_analyzer import analyze_scene
from plan10.lib.util import extract_frame

from plan10.lib.decomposer import decompose_scene

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))

if os.environ.get('ANIME','False') != 'False':
    from plan10.lib.anime_gen import GenerateImage, CreateBackground, CreateCharacterSheet
else:
    from plan10.lib.image_gen import GenerateImage, CreateBackground, CreateCharacterSheet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--ref', type=str, action='append', default=[])
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
    
    args = parser.parse_args()
    state_mgr = StateManager(args.output)
    first_run = False
    
    # Load or Initialize State
    if state_mgr.exists() and not args.reset:
        print("🔄 Resuming from state file...")
        state = state_mgr.load()
        refs = state['character_refs']
        visual_ids = state['visual_ids']
        beat_count = state['beat_count']
        current_media = state['current_media']
        story_context = state['story_context']
        history = state['history']
        pending_setup = state['pending_setup']
        needs_transition = state['needs_transition']
        video_queue = state.get('video_queue', [])
        scene_mode = state.get('scene_mode', False) 
        initial = state['initial_media']
        goal = state.get('goal')
        duration = state.get('duration')
    else:
        scene_mode = args.scene_mode
        first_run = True
        goal = args.goal
            
        print("🆕 Starting new loop...")
        refs = args.ref
        
        # Handle initial image generation if needed
        if not args.initial and not args.prompt:
            print("Error: --initial (-I) or --prompt (-P) required for a new run")
            sys.exit(-1)
            
        if not args.initial:
            GenerateImage(prompt=args.prompt, output=f'{args.output}/improv.png', width=args.width, height=args.height, seed=args.seed)
            initial = f'{args.output}/improv.png'
            current_media = initial
        else:
            _, image = extract_frame(args.initial, WIDTH, HEIGHT, 'last_frame.png')
            current_media = image
            initial = image

        if not refs:
            decompose_scene(
                input_image=current_media,
                output_dir=args.output,
                seed=args.seed
            )
            for p in ['character_1.png', 'character_2.png']:
                if Path(f'{args.output}/{p}').exists():
                    refs.append(f'{args.output}/{p}')
                    
        print(f"REFERENCES: {refs}")
        
        profiles = [CharacterProfile(ref) for ref in refs]
        visual_ids = []
        for profile in profiles:
            visual_ids.append(profile.get_visual_id(0))
        
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

    context = args.context
    if not context:
        context = analyze_scene(current_media)
    
    # Initialize Pipeline
    pipeline = Pipeline(refs, args.output, args.width, args.height, args.seed, visual_ids, 
        scene_mode=scene_mode, goal=goal)
    pipeline.initial_media = initial
    if first_run and not args.scene_mode:
        background = f'{args.output}/background.png'
        if Path(background).exists():
            current_media = pipeline.recreate_frame(background, context, 0)
        else:
            current_media = pipeline.recreate_frame(current_media, context, 0)
    
    # Execute ONE creative step
    try:
        result = pipeline.execute_step(
            current_media, context, beat_count, 
            history, pending_setup, needs_transition
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  Manual stop detected. Saving state...")
        state_mgr.save({
            "beat_count": beat_count, "current_media": current_media,
            "story_context": context, "history": history,
            "pending_setup": pending_setup, "needs_transition": needs_transition,
            "character_refs": refs, "visual_ids": visual_ids,
            "video_queue": video_queue,
            "output_dir": args.output, "width": args.width, "height": args.height, "seed": args.seed
        })
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(255)

    # Add the new video job to the queue
    video_queue.append(result['video_job'])
    
    # Clean up old completed jobs (keep last 3 for reference)
    video_queue = [job for job in video_queue if job['status'] in ['pending', 'processing']] + \
                  [job for job in video_queue if job['status'] == 'complete'][-3:]

    # Save state
    new_state = {
        "beat_count": result['beat_count'],
        "current_media": result['current_media'],
        "story_context": context,
        "history": result['history'][-3:],
        "pending_setup": result['pending_setup'],
        "needs_transition": result['needs_transition'],
        "character_refs": refs,
        "visual_ids": visual_ids,
        "video_queue": video_queue,
        "scene_mode": scene_mode,
        "output_dir": args.output,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "initial_media": initial,
        "goal": goal,
        "duration": args.duration,
    }
    state_mgr.save(new_state)
    
    print(f"✅ Beat {result['beat_count']} planned. Video queued for rendering.")
    sys.exit(result['beat_count'])

if __name__ == "__main__":
    main()