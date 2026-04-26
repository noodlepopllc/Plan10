#!/usr/bin/env python3
"""
Interactive Asset Generator CLI (v2.1)
Usage: python cli.py -c my_session.json
"""

import sys, os, json, time, argparse
from pathlib import Path

sys.path.append('./lib')
from brain import execute_task

# =============================================================================
# CONFIG & PATHS
# =============================================================================
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# CONTEXT MANAGEMENT
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
            
            # 🧹 Clean missing assets
            if isinstance(data.get("assets"), dict):
                cleaned, missing = {}, []
                for alias, info in data["assets"].items():
                    if info.get("path") and os.path.exists(info["path"]):
                        cleaned[alias] = info
                    else:
                        missing.append(alias)
                if missing:
                    print(f"🧹 Cleaned {len(missing)} missing assets from {path}")
                    data["assets"] = cleaned
                    save_context(path, data)
                return data
    return {"messages": [], "assets": {}, "history": []}

# =============================================================================
# HELPER: List outputs directory
# =============================================================================
def list_outputs():
    files = sorted([f.name for f in OUTPUT_DIR.iterdir() if f.is_file()])
    if not files:
        print("📂 outputs/ is empty.")
    else:
        print("📂 Files in outputs/:")
        for f in files:
            print(f"   📄 {f}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Interactive Asset Generator CLI")
    parser.add_argument('--context', '-c', type=str, default='context.json', 
                        help='Context file to load/save (default: context.json)')
    parser.add_argument('--max-steps', '-s', type=int, default=15, 
                        help='Max generation steps per task')
    args = parser.parse_args()

    context_path = args.context
    ctx = load_context(context_path)
    current_alias = None

    # Readline setup
    try:
        import readline
        histfile = Path.home() / f".asset_cli_{Path(context_path).stem}_history"
        if histfile.exists():
            readline.read_history_file(str(histfile))
    except ImportError:
        pass

    print("\n" + "="*60)
    print(" 🎨 INTERACTIVE ASSET GENERATOR v2.1")
    print(f" 📁 Context: {os.path.abspath(context_path)}")
    print(" Commands: /alias <name>, /import <alias> <filename>, /ls, /list, /clear, /remove <alias>, /reset, /quit")
    print(" Ctrl+C cancels current task | Ctrl+D quits & saves")
    print("="*60 + "\n")

    try:
        while True:
            alias_tag = f"[@{current_alias}] " if current_alias else ""
            try:
                line = input(f"🎯 {alias_tag}> ").strip()
            except EOFError:
                print("\n👋 Exiting. Context saved.")
                save_context(context_path, ctx)
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
                                "path": str(fpath.resolve()),
                                "type": asset_type,
                                "description": "Manually imported media",
                                "prompt": "N/A",
                                "metadata": {"source": "manual_import", "added": time.time()}
                            }
                            save_context(context_path, ctx)
                            print(f"✅ Imported '{arg1}' from outputs/{arg2}")
                        else:
                            print(f"❌ File not found in outputs/: {arg2}")
                            list_outputs()
                    else:
                        print("Usage: /import <alias> <filename>")
                        list_outputs()

                elif cmd == "ls":
                    list_outputs()

                elif cmd == "list":
                    assets = ctx.get("assets", {})
                    if not assets:
                        print("📦 No assets registered.")
                    else:
                        print("📦 Registered Assets:")
                        for a, info in assets.items():
                            exists = os.path.exists(info.get("path", ""))
                            print(f"  {a:20} {info.get('type','?'):8} {'✅' if exists else '❌'} | {info.get('path','')}")

                elif cmd == "clear":
                    ctx["messages"] = []
                    ctx["history"] = []
                    save_context(context_path, ctx)
                    print("🧹 Working memory cleared.")

                elif cmd == "remove" and arg1:
                    if arg1 in ctx.get("assets", {}):
                        ctx["assets"].pop(arg1)
                        save_context(context_path, ctx)
                        print(f"🗑️ Removed alias '{arg1}' from context.")
                    else:
                        print(f"❌ Alias '{arg1}' not found.")

                elif cmd == "reset":
                    ctx = {"messages": [], "assets": {}, "history": []}
                    save_context(context_path, ctx)
                    print(f"🗑️ Context '{context_path}' wiped. Starting fresh.")

                elif cmd in ("quit", "exit", "q"):
                    print("👋 Exiting. Context saved.")
                    save_context(context_path, ctx)
                    sys.exit(0)

                else:
                    print("❓ Unknown command. Use /alias, /import, /ls, /list, /clear, /remove, /reset, /quit")
                continue

            # === PROMPT EXECUTION ===
            target = current_alias

            # Skip if alias exists & file is valid
            if target and target in ctx.get("assets", {}):
                info = ctx["assets"][target]
                if info.get("path") and os.path.exists(info["path"]):
                    print(f"⏭️  Asset [{target}] already exists. Use /alias to change target or /remove to clear.")
                    continue
                else:
                    print(f"🔄 Stale reference for [{target}] found. Will regenerate.")
                    ctx["assets"].pop(target, None)

            # Prepare context
            if ctx.get("messages"):
                ctx["history"].extend(ctx["messages"])
                ctx["messages"] = []

            ctx["target_alias"] = target
            ctx["messages"].append({"role": "user", "content": line})
            save_context(context_path, ctx)

            print(f"\n🚀 Running [{target or 'Unnamed'}]: {line[:60]}...")
            try:
                ctx, log = execute_task(line, max_steps=args.max_steps, target_alias=target, initial_ctx=ctx)
                last_step = log["completed_steps"][-1] if log.get("completed_steps") else "Done"
                print(f"✅ Completed: {last_step}")
            except KeyboardInterrupt:
                print("\n⛔ Task cancelled by user.")
            except Exception as e:
                print(f"❌ Execution failed: {e}")
            
            save_context(context_path, ctx)

            # Auto-clear alias after generation to prevent accidental overwrites
            if current_alias:
                current_alias = None
                print("🏷️  Target alias auto-cleared.")
            print("-" * 60)

    finally:
        try:
            import readline
            histfile = Path.home() / f".asset_cli_{Path(context_path).stem}_history"
            readline.write_history_file(str(histfile))
        except:
            pass

if __name__ == "__main__":
    main()
