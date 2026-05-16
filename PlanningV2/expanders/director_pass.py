#!/usr/bin/env python3
"""
Takes raw dialog JSON + story context + character registry.
Runs director enrichment pass. Outputs enriched_sequence.json ready for prompt assembly.
"""
import sys, json, subprocess, os, re

def call_llm(prompt, sys_file, retries=2):
    for attempt in range(retries + 1):
        res = subprocess.run(
            ["python", "lib/qwen_llm.py", "-S", sys_file, "-P", prompt],
            capture_output=True, text=True, check=True
        )
        raw = re.sub(r'```json|```', '', res.stdout.strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == retries:
                raise RuntimeError("LLM failed to output valid JSON after retries")

def main():
    output_dir = sys.argv[1]
    raw_json_path = f"{output_dir}/sequence_dialog.json"  # or wherever your raw beats live
    sys_prompt = "PlanningV2/prompts/sys_director_enrich.txt"
    
    with open(raw_json_path) as f:
        raw_data = json.load(f)
    
    story_context = raw_data.get("metadata", {}).get("story_context", "")
    registry_note = "CHARACTER REGISTRY: the_sorceress = ornate gown, sharp features, pale porcelain skin, silver-streaked dark hair, tall commanding posture. the_princess = tattered linen rags, youthful feminine frame, dark-blonde hair, pale skin, restrained posture."
    
    prompt = f"{registry_note}\n\nSTORY CONTEXT: {story_context}\n\nRAW BEATS:\n{json.dumps(raw_data.get('beats', raw_data), indent=2)}"
    
    enriched = call_llm(prompt, sys_prompt)
    
    out_path = f"{output_dir}/enriched_sequence.json"
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2)
    print(f"✓ Director pass complete: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: director_pass.py <output_dir>", file=sys.stderr)
        sys.exit(1)
    main()