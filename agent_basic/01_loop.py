"""
01_loop.py  —  一個 agent 最小只需要這些

讀書會導讀順序：先看最下面的 while 迴圈，再往上看 TOOLS 和 run_tool。
跑法：  uv run python 01_loop.py "What OS is this and how many .py files are here?"
"""
import json, os, subprocess, sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()                                   # .env 裡放 OPENROUTER_API_KEY / OPENROUTER_MODEL
client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3")

# ── 1. 工具契約：告訴模型「你可以呼叫什麼」。這是 JSON Schema，模型只看得到這個 ──
TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]

# ── 2. 工具實作：模型說要呼叫 run_shell，真正動手的是這裡。名字要跟上面對得上 ──
def run_tool(name: str, args: dict) -> str:
    if name == "run_shell":
        r = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout + r.stderr
    return f"unknown tool: {name}"

# ── 3. 對話歷史：agent 的「記憶」就是這個 list。每一輪都把整份送給模型 ──
messages = [
    {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
    {"role": "user", "content": " ".join(sys.argv[1:]) or "What OS is this?"},
]

# ── 4. 迴圈：問模型 → 模型要嘛回答、要嘛要求用工具 → 用完把結果塞回歷史 → 再問 ──
for step in range(10):                          # 上限：避免模型卡住一直叫工具
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    msg = response.choices[0].message
    messages.append(msg)                        # 模型說的話也是歷史的一部分（含它的 tool_calls）

    if not msg.tool_calls:                      # 沒有要用工具 = 它認為答完了
        print(f"\n[answer after {step + 1} round(s)]\n{msg.content}")
        break

    for call in msg.tool_calls:                 # 一輪可能同時要求多個工具
        args = json.loads(call.function.arguments)
        print(f"🔧 {call.function.name}({args})")
        result = run_tool(call.function.name, args)
        messages.append({                       # 工具結果要用 role=tool 回去，並對上 tool_call_id
            "role": "tool",
            "tool_call_id": call.id,
            "content": result[:4000],           # 粗略截斷，避免把整個檔案塞進 context
        })