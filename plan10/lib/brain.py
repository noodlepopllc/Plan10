from pathlib import Path
import json, os, gc, torch, re 
from plan10.lib.tools import ToolHandler
from plan10.lib.qwen_llm import llm_chat

CONFIG_FILE = "config.json"

def system_prompt(fn='system/bot.txt'):
    if not os.path.exists(fn):
        repo_root = Path(__file__).parent.parent
        fn = repo_root / "system" / Path(fn).name
    prompt = Path(fn).read_text()
    while prompt:
        yield [{"role": "system", "content": prompt}]
    return None

system_prompt = system_prompt() 

def _strip_thinking(raw):
    """Extract thinking blocks from Qwen response."""
    match = re.search(r'<think>(.*?)</think>', raw, flags=re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        response = raw.replace(match.group(0), '').strip()
        return thinking, response
    return "", raw.strip()
     
# =============================================================================
# PARSING
# =============================================================================

import json
import re

def parse_tool_response(response_json={}, raw_content=""):
    """
    Parses tool calls from Ollama's structured response.
    Falls back to Qwen XML format if the model outputs it in text instead.
    Returns: [{"name": str, "arguments": dict}, ...]
    """
    # 1️⃣ Try Ollama's structured tool_calls first
    tool_calls = response_json.get("tool_calls", [])
    
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_raw = func.get("arguments", "{}")
            
            # Ollama sometimes returns args as dict, sometimes as JSON string
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_raw if isinstance(args_raw, dict) else {}
                
            return {"name": name, "arguments": args}

    # 2️⃣ Fallback: Qwen XML format (if model outputs it in content instead)
    if raw_content:
        func_match = re.search(r'<function=(.*?)>', raw_content)
        if func_match:
            func_name = func_match.group(1).strip()
            params = {}
            for match in re.finditer(r'<parameter=(.*?)>(.*?)</parameter>', raw_content, re.DOTALL):
                n, v = match.groups()
                n, v = n.strip(), v.strip()
                try:
                    params[n] = json.loads(v) if v.startswith(('[', '{')) else v
                except:
                    params[n] = v
            return {"name": func_name, "arguments": params}

    return {}

# =============================================================================
# TASK EXECUTOR
# =============================================================================
def execute_task(task_description, max_steps=15, target_alias=None, initial_ctx=None):
    ctx = initial_ctx
    ctx["target_alias"] = target_alias  # Fallback if LLM forgets to pass alias
    toolhandler = ToolHandler()
    
    messages = [{"role": "user", "content": [{"type":"text", "text": f"TASK: {task_description}"}]}]
    task_state = {"goal": task_description, "completed_steps": [], "assets_created": []}
    
    for step in range(1, max_steps + 1):
        print(f"\n━━━ STEP {step}/{max_steps} ━━━")
        
        # Inject live state (temporary, removed after generation)
        state_msg = f"CURRENT STATE:\n📦 Assets:\n{toolhandler.render_assets(ctx)}\n📋 Goal: {task_description}"
        messages.append({"role": "user", "content": [{"type": "text", "text": state_msg}]})
        
        response = llm_chat(messages, tools=ToolHandler.TOOLS, enable_thinking=False)
        response_clean = response.get('response_clean','')
        thinking = response.get('thinking', '')
        
        if thinking:
            print("🤔 THINKING:\n" + "─" * 50)
            print(thinking)
            print("─" * 50)
        print("📝 RESPONSE:", response_clean[:300])
        messages.pop()  # Remove injected state message
        
        # Parse tool call
        if response_clean:
            tool_payload = parse_tool_response(raw_content=response_clean)
        else:
            tool_payload = parse_tool_response(response_json=response)
        if not tool_payload:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response_clean}]})
            messages.append({"role": "user", "content": [{"type": "text", "text": "Call a tool to proceed."}]})
            continue
            
        # 🔑 ALIAS FLOW: LLM passes alias in args -> run_tool pops it -> returns it in result
        tool_payload['arguments']['alias'] = target_alias
        result = toolhandler.run_tool(tool_payload["name"], tool_payload["arguments"], ctx)
        
        # 🔑 EXIT CONDITION: Asset successfully created & registered
        created_alias = result.get("asset_alias")
        if created_alias:
            asset_type = ctx["assets"].get(created_alias, {}).get("type", "unknown")
            task_state["assets_created"].append(f"{created_alias} ({asset_type})")
            task_state["completed_steps"].append(f"Step {step}: {tool_payload['name']} → {created_alias}")
            print(f"\n🎯 ASSET CREATED: {created_alias} [{asset_type}]")
            print("✅ TASK COMPLETE - Exiting loop.")
            break
            
        # Handle errors or continue (analysis, retries, etc.)
        messages.append({"role": "assistant", "content": [{"type":"text","text": response_clean}]})
        feedback = result.get("message", "Tool executed but no asset created. Adjust and retry.")
        messages.append({"role": "tool", "content": [{"type": "text", "text": f"[TOOL RESULT] {feedback}"}]})
        
    print(f"\n📊 SUMMARY: Ran {step} steps. Assets: {task_state['assets_created']}")
    return ctx, task_state
