# AI Agent 實作 Resources

## Knowledge

- [OpenRouter — Tool & Function Calling](https://openrouter.ai/docs/guides/features/tool-calling)
  跟 agent.py 用的 `chat.completions` 格式完全對應：`tools` 參數、回應中的 `tool_calls`、`role: "tool"` + `tool_call_id` 回傳結果、「直到沒有 tool_calls 才停」的迴圈（範例上限 10 輪）。Use for: 迴圈與訊息格式的權威來源；查哪些模型支援 tools（`openrouter.ai/models?supported_parameters=tools`）。
- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
  Agent 的一句話定義：「LLMs using tools based on environmental feedback in a loop」；workflow vs agent；工具介面（ACI）設計建議；停止條件。Use for: 第一課的定義、工具設計那一課的「描述要讓模型看得懂」。
- [Anthropic — Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
  每一輪送進模型的 context 要精簡；system prompt 的「高度」；工具結果要省 token；compaction 與 note-taking（TODO 清單就是一種）。Use for: 記憶、system hint、TODO 那幾課的「為什麼要這樣做」。
- [《深入理解 AI Agent》— chapter2/system-hint 實驗（原始碼來源）](https://github.com/bojieli/ai-agent-book/tree/main/chapter2/system-hint)
  agent.py 的原出處，書中「實驗 2-9：Agent 狀態欄（system hint）」。README 解釋為什麼在 context 末尾用一條 `role=user` 注入狀態摘要。Use for: system hint 那一課的設計動機；本機副本在 `~/tutorials/ai-agent-book/chapter2/system-hint/`。
- [OpenAI — Function calling guide](https://developers.openai.com/api/docs/guides/function-calling)
  注意：現在以 Responses API 寫（`function_call` / `function_call_output`），名詞跟 agent.py 不同。Use for: 補充 `tool_choice`、parallel tool calls 的概念，不要拿來對照程式碼。
- [pydantic — Models](https://docs.pydantic.dev/latest/concepts/models/)
  `BaseModel`、`Field(default_factory)`、`model_dump(mode="json")`、`model_json_schema()`。Use for: 工具那一課提「契約可以從 pydantic 產生」時查語法。

## Wisdom (Communities)

- 讀書會本身：facilitator 帶讀；每課結尾的問題帶回來問 agent（我）。
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) / [OpenRouter Discord](https://discord.gg/openrouter)
  Use for: 哪些模型 tool calling 穩、provider 錯誤率。（使用者尚未表達是否想加入社群。）

## Gaps
- 尚缺一份「tool calling 訊息格式」的繁體中文一手資料；目前以 OpenRouter 英文文件為準。
- 本機 sandbox / 權限控管的入門資源（0006 保護那一課若要延伸再找）。
