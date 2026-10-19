# 起點：facilitator 已熟 agent.py，學員從 tool calling 零基礎起算

使用者（facilitator）已完整讀過並改過 `agent.py`（pydantic 化、log 清理、.env 模型切換），
並自己寫出 60 行的 `01_loop.py` 最小版且跑通。學員是讀書會成員，假設會 Python、
呼叫過一次 chat API，但不假設看過 `tools` / `tool_calls` / `role: "tool"`。

## Implications
- Lesson 不需要教 Python 或 API 基本呼叫；第一課直接從 tool calling 的三條規則開始。
- 使用者明講「不需要實作單元」，回饋一律用預測行為的小測驗，不留程式填空。
- 每課要能對應回 VSCode 裡的函式名；行號只當輔助。
