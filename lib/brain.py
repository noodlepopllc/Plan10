from pathlib import Path
import json, os, gc, torch, re 
from tools import ToolHandler
from qwen_llm import llm_chat

CONFIG_FILE = "config.json"
os.environ['TRANSFORMERS_OFFLINE'] = "1"

def system_prompt(fn='system/bot.txt'):
    prompt = Path(fn).read_text()
    while prompt:
        yield [{"role": "system", "content": prompt}]
    return None

system_prompt = system_prompt() 


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4"
    )

def load_config():
    # ⚠️ MUST point to a VL model: e.g. "Qwen/Qwen3-VL-8B-Instruct"
    cfg = {} 
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f: cfg.update(json.load(f))
        except: print(f'{CONFIG_FILE} is missing or broken')
    return cfg

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
def parse_tool_call(raw_text):
    """Parse Qwen XML tool format."""
    func_match = re.search(r'<function=(.*?)>', raw_text)
    if not func_match:
        return None
    
    func_name = func_match.group(1).strip()
    params = {}
    
    for match in re.finditer(r'<parameter=(.*?)>(.*?)</parameter>', raw_text, re.DOTALL):
        name, value = match.groups()
        name = name.strip()
        value = value.strip()
        
        if value.startswith('[') or value.startswith('{'):
            try:
                params[name] = json.loads(value)
            except:
                params[name] = value
        else:
            params[name] = value
    
    return {"name": func_name, "arguments": params}

# =============================================================================
# TASK EXECUTOR
# =============================================================================
def execute_task(task_description, max_steps=15, target_alias=None, initial_ctx=None):
    ctx = initial_ctx or load_context()
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
        response_clean = response['response_clean']
        thinking = response.get('thinking', '')
        
        if thinking:
            print("🤔 THINKING:\n" + "─" * 50)
            print(thinking)
            print("─" * 50)
        print("📝 RESPONSE:", response_clean[:300])
        messages.pop()  # Remove injected state message
        
        # Parse tool call
        tool_payload = parse_tool_call(response_clean)
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
