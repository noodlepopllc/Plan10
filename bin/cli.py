#!/usr/bin/env python3
"""
Interactive Asset Generator CLI (v3.0)
Adds /cp <alias> <new_dir> for scene branching (e.g., outfit changes).
Keeps relative paths. Enforces OUTPUT_DIR = parent of context.
Usage: python cli.py -c alice/context.json
"""

import sys, os, json, time, argparse, shutil
from pathlib import Path

sys.path.append('./lib')
from config import load_environ

load_environ()
os.environ['BATCH'] = 'False'

from brain import execute_task

# =============================================================================
# CONTEXT HELPERS
# =============================================================================
def save_context(path, ctx):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ctx, f, indent=2)

def load_context(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
            data.setdefault("messages", [])
            data.setdefault("assets", {})
            data.setdefault("history", [])
            return data
    return {"messages": [], "assets": {}, "history": []}

# =============================================================================
# PATH RESOLUTION HELPER (Relative-First)
# =============================================================================
def asset_exists(rel_path, out_dir):
    """Checks if asset exists relative to OUTPUT_DIR, then falls back to CWD"""
    p = Path(rel_path)
    return (out_dir / p).exists() or p.exists()

# =============================================================================
# CLI LOOP
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Interactive Asset Generator CLI")
    parser.add_argument('--context', '-c', type=str, default=None, help='Path to context file')
    parser.add_argument('--max-steps', '-s', type=int, default=5, help='Max generation steps per task')
    args = parser.parse_args()

    # 1️⃣ RESOLVE CONTEXT PATH
    if args.context:
        ctx_path = Path(args.context).resolve()
    elif os.environ.get('CONTEXT_FILE'):
        ctx_path = Path(os.environ['CONTEXT_FILE']).resolve()
    else:
        ctx_path = Path("outputs/context.json").resolve()

    # 2️⃣ DERIVE OUTPUT_DIR STRICTLY AS PARENT
    OUTPUT_DIR = ctx_path.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3️⃣ SYNC ENVIRONMENT FOR DOWNSTREAM TOOLS
    os.environ['CONTEXT_FILE'] = str(ctx_path)
    os.environ['OUTPUT_DIR'] = str(OUTPUT_DIR)

    # 4️⃣ LOAD & VALIDATE CONTEXT (Interactive Cleanup)
    ctx = load_context(ctx_path)
    
    if isinstance(ctx.get("assets"), dict):
        missing = []
        for alias, info in list(ctx["assets"].items()):
            raw_path = info.get("path", "")
            if not raw_path:
                missing.append(alias)
                continue

            if not asset_exists(raw_path, OUTPUT_DIR):
                missing.append(alias)

        if missing:
            print(f"\n⚠️  Found {len(missing)} missing/stale assets: {', '.join(missing)}")
            if input("Clean these from context? [y/N]: ").strip().lower() in ('y', 'yes'):
                for alias in missing:
                    ctx["assets"].pop(alias, None)
                print("✅ Cleaned missing assets.")
                save_context(ctx_path, ctx)

    current_alias = None

    # Readline setup
    try:
        import readline
        histfile = Path.home() / f".asset_cli_{ctx_path.stem}_history"
        if histfile.exists():
            readline.read_history_file(str(histfile))
    except ImportError:
        pass

    print("\n" + "="*60)
    print(" 🎨 INTERACTIVE ASSET GENERATOR v3.0")
    print(f" 📁 Output Dir : {OUTPUT_DIR}")
    print(f" 📄 Context    : {ctx_path}")
    print(" Commands: /alias, /cp <alias> <new_dir>, /import <alias> <file>, /ls, /list, /clear, /remove, /reset, /quit")
    print(" Ctrl+C cancels current task | Ctrl+D quits & saves")
    print("="*60 + "\n")

    def list_outputs():
        files = sorted([f.name for f in OUTPUT_DIR.iterdir() if f.is_file()])
        if not files:
            print("📂 Output directory is empty.")
        else:
            print(f"📂 Files in {OUTPUT_DIR.name}/:")
            for f in files:
                print(f"   📄 {f}")

    try:
        while True:
            alias_tag = f"[@{current_alias}] " if current_alias else ""
            try:
                line = input(f"🎯 {alias_tag}> ").strip()
            except EOFError:
                print("\n👋 Exiting. Context saved.")
                save_context(ctx_path, ctx)
                sys.exit(0)
            except KeyboardInterrupt:
                print("\n⛔ Interrupted.")
                continue

            if not line:
                continue

            # === SLASH COMMANDS ===
            if line.startswith("/"):
                parts = line[1:].strip().split(None, 3)
                cmd = parts[0].lower()
                arg1 = parts[1] if len(parts) > 1 else None
                arg2 = parts[2] if len(parts) > 2 else None

                if cmd == "alias":
                    current_alias = arg1.strip() if arg1 else None
                    print(f"🏷️  Target alias set to: {current_alias}" if current_alias else "🏷️  Target alias cleared.")
                        
                elif cmd == "import":
                    if arg1 and arg2:
                        fpath = OUTPUT_DIR / arg2
                        if fpath.is_file():
                            ext = fpath.suffix.lower()
                            asset_type = "image" if ext in {'.png','.jpg','.jpeg','.webp'} else "video" if ext in {'.mp4','.mov','.webm'} else "file"
                            ctx.setdefault("assets", {})[arg1] = {
                                "path": arg2,  # RELATIVE ONLY
                                "type": asset_type,
                                "description": "Manually imported media",
                                "prompt": "N/A",
                                "metadata": {"source": "manual_import", "added": time.time()}
                            }
                            save_context(ctx_path, ctx)
                            print(f"✅ Imported '{arg1}' as '{arg2}'")
                        else:
                            print(f"❌ File not found in {OUTPUT_DIR.name}/: {arg2}")
                            list_outputs()
                    else:
                        print("Usage: /import <alias> <filename>")
                        list_outputs()

                # 🆕 NEW: Copy Asset to New Scene
                elif cmd == "cp":
                    if arg1 and arg2:
                        alias, new_dir = arg1, arg2
                        if alias not in ctx.get("assets", {}):
                            print(f"❌ Alias '{alias}' not found.")
                        else:
                            src_info = ctx["assets"][alias]
                            src_rel = src_info.get("path", "")
                            
                            # Resolve source
                            src_file = Path(src_rel)
                            if not src_file.exists():
                                print(f"❌ Source file missing: {src_file}")
                            else:
                                # Prepare destination
                                dest_dir = Path(new_dir)
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                dest_file = dest_dir / src_file.name
                                
                                if dest_file.exists():
                                    print(f"⚠️ Overwriting existing file: {dest_file}")
                                
                                # Copy file (preserves timestamps)
                                shutil.copy2(src_file, dest_file)
                                
                                # Create/Update context in new dir
                                dest_ctx_path = dest_dir / "context.json"
                                if dest_ctx_path.exists():
                                    new_ctx = load_context(str(dest_ctx_path))
                                else:
                                    new_ctx = {"messages": [], "assets": {}, "history": []}
                                    
                                # Add asset with RELATIVE path (filename only)
                                new_ctx["assets"][alias] = {
                                    "path": str(dest_file),
                                    "type": src_info.get("type", "unknown"),
                                    "description": src_info.get("description", "Copied asset"),
                                    "prompt": src_info.get("prompt", ""),
                                    "metadata": {
                                        "source": f"copied from {alias} ({src_rel})",
                                        "added": time.time()
                                    }
                                }
                                save_context(str(dest_ctx_path), new_ctx)
                                
                                print(f"✅ Copied '{alias}' -> '{dest_dir}/{src_file.name}'")
                                print(f"   Context created/updated: {dest_ctx_path}")
                                print("   You can now edit the asset in the new directory.")
                    else:
                        print("Usage: /cp <alias> <new_directory>")

                elif cmd == "ls":
                    list_outputs()

                elif cmd == "list":
                    assets = ctx.get("assets", {})
                    if not assets:
                        print("📦 No assets registered.")
                    else:
                        print("📦 Registered Assets:")
                        for a, info in assets.items():
                            exists = asset_exists(info.get("path", ""), OUTPUT_DIR)
                            print(f"  {a:20} {info.get('type','?'):8} {'✅' if exists else '❌'} | {info.get('path','')}")

                elif cmd == "clear":
                    ctx["messages"] = []
                    ctx["history"] = []
                    save_context(ctx_path, ctx)
                    print("🧹 Working memory cleared.")

                elif cmd == "remove" and arg1:
                    if arg1 in ctx.get("assets", {}):
                        ctx["assets"].pop(arg1)
                        save_context(ctx_path, ctx)
                        print(f"🗑️ Removed alias '{arg1}' from context.")
                    else:
                        print(f"❌ Alias '{arg1}' not found.")

                elif cmd == "reset":
                    ctx = {"messages": [], "assets": {}, "history": []}
                    save_context(ctx_path, ctx)
                    print(f"🗑️ Context '{ctx_path.name}' wiped. Starting fresh.")

                elif cmd in ("quit", "exit", "q"):
                    print("👋 Exiting. Context saved.")
                    save_context(ctx_path, ctx)
                    sys.exit(0)

                else:
                    print("❓ Unknown command. Use /alias, /cp, /import, /ls, /list, /clear, /remove, /reset, /quit")
                continue

            # === PROMPT EXECUTION ===
            target = current_alias

            if target and target in ctx.get("assets", {}):
                info = ctx["assets"][target]
                raw_path = info.get("path", "")
                if raw_path and asset_exists(raw_path, OUTPUT_DIR):
                    print(f"⏭️  Asset [{target}] already exists. Use /alias to change target or /remove to clear.")
                    continue
                else:
                    print(f"🔄 Stale reference for [{target}] detected. Will regenerate.")
                    ctx["assets"].pop(target, None)

            if ctx.get("messages"):
                ctx["history"].extend(ctx["messages"])
                ctx["messages"] = []

            ctx["target_alias"] = target
            ctx["messages"].append({"role": "user", "content": line})
            save_context(ctx_path, ctx)

            print(f"\n🚀 Running [{target or 'Unnamed'}]: {line[:60]}...")
            try:
                ctx, log = execute_task(line, max_steps=args.max_steps, target_alias=target, initial_ctx=ctx)
                last_step = log["completed_steps"][-1] if log.get("completed_steps") else "Done"
                print(f"✅ Completed: {last_step}")
            except KeyboardInterrupt:
                print("\n⛔ Task cancelled by user.")
            except Exception as e:
                print(f"❌ Execution failed: {e}")
            
            save_context(ctx_path, ctx)

            if current_alias:
                current_alias = None
                print("🏷️  Target alias auto-cleared.")
            print("-" * 60)

    finally:
        try:
            import readline
            readline.write_history_file(str(histfile))
        except:
            pass

if __name__ == "__main__":
    main()