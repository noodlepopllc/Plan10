import sys
import argparse
from pathlib import Path
import os

from plan10.lib.config import load_environ
load_environ()

WGP = os.environ.get("WGP","False") != "False"
LTX = os.environ.get("LTX","False") != "False"
MMH3 = os.environ.get('MMH3', 'False') != 'False'

MMH3_Prompt = '''You enhance rough video prompts into structured audiovisual rewrite prompts for I2VA (first-frame image → video).

Hard rule: NEVER paraphrase or narrate these instructions in the output. Do not explain the format or summarize the user prompt as a story synopsis. Emit the alignment line exactly once as the first line, then write only concrete audiovisual scene content.

Output rules:
1) First line must be exactly:
   For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
   Then one blank line.
2) Then output exactly these three fields in order — always all three; never stop after the description alone:
   integrated_multimodal_description:
   overall_soundscape:
   non_diegetic_music:
3) Write the body in English. Preserve original language only inside <d> dialogue/lyrics and for on-screen text in double quotes.
4) [Shot 1] has no timestamp. Later shots use: [Shot N] At MM:SS.mmm, ...
5) Camera motion is natural English with motion type and, when meaningful, amplitude (with small/large amplitude) and speed (at slow/fast speed).
6) Speakers use stable token IDs with parentheses: (S1), (S2). You must map them directly to their exact positions in the opening line of the description (e.g., "<Picture 1> is the first frame, where the subject on the left is assigned to token ID (S1), and the subject on the right is assigned to token ID (S2)"). Dialogue must NEVER be placed on a standalone line or appended to the end of the text. It must be woven directly inside the action sentence describing the speaker's lip movements using the format: saying in an on-screen voice [Language] <d>"exact words"</d>. Voiceover uses "says in an off-screen voiceover" and notes lips remain closed.
7) overall_soundscape: 1–4 English sentences on ambience, physical action sounds, non-verbal human sounds. No dialogue/singing/diegetic music. Use N/A only for total silence.
8) non_diegetic_music: 1–3 sentences on instrumentation, tempo, dynamics only (no abstract mood words). Use N/A when absent.
9) Picture 1 is the first frame of Shot 1. Minimize descriptive tokens for static visual attributes (clothing colors, hair styles) already present in <Picture 1>. Open [Shot 1] by establishing the mapping contract between spatial positions and token IDs, then immediately transition with a clean motion trigger: "Breaking their initial layout, the characters activate smoothly into motion with no identity drift."
10) You must budget the motion chronologically using explicit sequential time intervals formatted strictly in standard MM:SS.mmm timecode (e.g., "From 00:00.000 to 00:03.000...", "From 00:03.000 to 00:06.000..."). For each time block, assign exactly ONE primary subject action mapped directly to an established token ID like (S1) or (S2) with parentheses, and ONE camera movement. The dialogue tag string must sit directly inside the specific timestamp block where it is being actively spoken by the character token. Define exactly when spoken dialogue ends and when lips close to halt motion. End the description block with the phrase: "Strict identity preservation is maintained for (S1) and (S2) throughout the clip."
"Hard Constraint: The total duration of the script must never exceed 00:05.000 seconds. You must compress the entire action narrative into a maximum of two sequential time blocks that fit perfectly inside a 5-second window (e.g., Block 1: 00:00.000 to 00:03.000, Block 2: 00:03.000 to 00:05.000)."
'''

if WGP:
    from plan10.lib.wgp import GenerateVideo
elif LTX:
    from plan10.lib.ltx import GenerateVideo
elif MMH3:
    from plan10.lib.mmh3 import GenerateVideo
else:
    from plan10.lib.image_to_video import GenerateVideo

from plan10.emergent.state_manager import StateManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output', type=str, default="feedback_output")
    args, _ = parser.parse_known_args()
    
    state_mgr = StateManager(args.output)
    
    if not state_mgr.exists():
        print("No state file found. Nothing to render.")
        sys.exit(0)
    
    state = state_mgr.load()
    video_queue = state.get('video_queue', [])
    duration = int(state.get('duration', '5'))
    
    # Find the first pending job
    pending_job = None
    for job in video_queue:
        if job['status'] == 'pending':
            pending_job = job
            break
    
    if not pending_job:
        print("No pending video jobs.")
        sys.exit(0)
    
    print(f"\n🎬 Rendering beat {pending_job['beat']}...")
    print(f"Input: {pending_job['input_media']}")
    print(f"Output: {pending_job['output_path']}")
    print(f"Prompt: {pending_job['prompt'][:100]}...")
    
    # Mark as processing
    pending_job['status'] = 'processing'
    state_mgr.save(state)
    
    try:
        prompt = pending_job['prompt']

        # Generate the video
        GenerateVideo(
            prompt=prompt,
            media=pending_job['input_media'],
            output=pending_job['output_path'],
            duration_sec=duration,
            seed=pending_job['seed']
        )
        
        # Mark as complete and update current_media
        pending_job['status'] = 'complete'
        state['current_media'] = pending_job['output_path']
        state_mgr.save(state)
        
        print(f"✅ Beat {pending_job['beat']} rendered successfully.")
        sys.exit(pending_job['beat'])  # Positive = success
        
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        pending_job['status'] = 'failed'
        state_mgr.save(state)
        sys.exit(255)

if __name__ == "__main__":
    main()