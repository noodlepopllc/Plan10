from pathlib import Path
import torch, os, sys, gc, json, re, time, argparse
sys.path.append('./lib')
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from brain import execute_task
from config import load_environ

load_environ()

# =============================================================================
# CONTEXT
# =============================================================================
def save_context(ctx):
    CONTEXT_FILE = Path(os.environ['CONTEXT_FILE']) if 'CONTEXT_FILE' in os.environ else Path("context.json")
    with open(CONTEXT_FILE, "w") as f:
        json.dump(ctx, f, indent=2)

def load_context():
    CONTEXT_FILE = Path(os.environ['CONTEXT_FILE']) if 'CONTEXT_FILE' in os.environ else Path("context.json")
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE) as f:
            data = json.load(f)
            data.setdefault("messages", [])
            data.setdefault("assets", {})
            data.setdefault("history", [])
            
            # 🧹 Remove assets that no longer exist on disk
            if isinstance(data.get("assets"), dict):
                missing = []
                cleaned = {}
                for alias, info in data["assets"].items():
                    path = info.get("path", "")
                    if path and os.path.exists(path):
                        cleaned[alias] = info
                    else:
                        missing.append(alias)
                
                if missing:
                    print(f"🧹 Cleaned up {len(missing)} missing assets: {', '.join(missing)}")
                    data["assets"] = cleaned
                    save_context(data)
            
            return data
            
    return {"messages": [], "assets": {}, "history": []}


# =============================================================================
# MAIN (Adapted to your alias/multi-task workflow)
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('prompt', nargs='?', default="Generate a cyberpunk city at night.")
    p.add_argument('--max-steps', '-s', type=int, default=5)
    p.add_argument('--context', '-K', type=str, default=None, help='Keep existing context/assets')
    p.add_argument('-F', '--fileprompt', action='store_true')
    args = p.parse_args()

    CONTEXT_FILE = args.context

    if args.fileprompt:
        OUTPUT_DIR = args.prompt.replace('.txt','')
    else:
        OUTPUT_DIR = "outputs"

    CONTEXT_FILE = args.context if args.context else f"{OUTPUT_DIR}/context.json"

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(exist_ok=True)

    os.environ['CONTEXT_FILE'] = CONTEXT_FILE
    os.environ['OUTPUT_DIR'] = OUTPUT_DIR


    # Reset context unless -K is passed
    if  os.path.exists(CONTEXT_FILE) and (not args.fileprompt and not args.context):
        os.remove(CONTEXT_FILE)

    raw = Path(args.prompt).read_text() if args.fileprompt else args.prompt
    prompt_list = [p.strip() for p in raw.split('\n\n') if p.strip()]

    ctx = load_context()
    if "history" not in ctx: ctx["history"] = []

    # 🔍 FILTER & ALIAS PARSING
    tasks_to_run = []
    for prompt_text in prompt_list:
        target_alias = None
        clean_prompt = prompt_text
        
        if prompt_text.startswith(">> ALIAS:"):
            lines = prompt_text.split('\n', 1)
            target_alias = lines[0].replace(">> ALIAS:", "").strip()
            clean_prompt = lines[1].strip() if len(lines) > 1 else ""

        # Skip if alias exists AND physical file is valid
        if target_alias and target_alias in ctx.get("assets", {}) and os.path.exists(ctx["assets"][target_alias]["path"]):
            print(f"⏭️ Skipping [{target_alias}]: Asset already exists at {ctx['assets'][target_alias]['path']}")
            continue
        else:
            if target_alias and target_alias in ctx.get("assets"):
                ctx["assets"].pop(target_alias)  # Clean stale ref
            
        tasks_to_run.append((clean_prompt, target_alias))

    if not tasks_to_run:
        print("✅ All requested assets already exist on disk. Exiting.")
        sys.exit(0)

    # EXECUTE CHAIN
    for i, (clean_prompt, target_alias) in enumerate(tasks_to_run, 1):
        # 1. Archive History & Clear Working Memory
        if ctx.get("messages"):
            ctx["history"].extend(ctx["messages"])
            ctx["messages"] = []

        # 2. Create TEMPORARY scoped context for this task
        task_ctx = {**ctx}  # Copy master state
        current_assets = ctx.get("assets", {})
        clean_lower = clean_prompt.lower()
        import re
        relevant_assets = {
            k: v for k, v in current_assets.items()
            if re.search(rf'\b{re.escape(k.lower())}\b', clean_lower) or k == target_alias or k.startswith('bg') or k.startswith('char')
        }

        print("RELEVANT ASSETS: ", [x for x in relevant_assets.keys()])

        # Inject available list + scoped assets into temp context
        task_ctx["assets"] = relevant_assets
        task_ctx["target_alias"] = target_alias
        task_ctx["messages"] = [{"role": "user", "content": f"Available assets: {', '.join(relevant_assets.keys())}\n\n{clean_prompt}"}]

        print(f"\n🚀 [{i}/{len(tasks_to_run)}] Running [{target_alias}] (scoped: {len(relevant_assets)} assets): {clean_prompt[:60]}...")

        # 3. Execute
        result_ctx, log = execute_task(clean_prompt, max_steps=args.max_steps, target_alias=target_alias, initial_ctx=task_ctx)

        # 4. Merge ONLY the newly created asset back into MASTER context
        new_assets = result_ctx.get("assets", {})
        for alias, info in new_assets.items():
            ctx["assets"][alias] = info  # Add or update without dropping other assets

        ctx["messages"] = result_ctx.get("messages", [])
        ctx["history"] = result_ctx.get("history", ctx["history"])
        save_context(ctx)

    path = Path("./logs")
    path.mkdir(parents=True, exist_ok=True)
    log_path = f"./logs/task_log_{int(time.time())}.json"
    with open(log_path, "w") as f:
        json.dump({"tasks_run": len(tasks_to_run), "final_assets": ctx["assets"]}, f, indent=2)
    print(f"📄 Log saved: {log_path}")