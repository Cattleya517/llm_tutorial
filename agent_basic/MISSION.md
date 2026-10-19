# Mission: 讀懂一個 AI agent 的實作（agent_basic/agent.py）

## Why
讀書會要帶大家在 VSCode 裡一起看 `agent.py`（約 1000 行的 SystemHintAgent），
目標不是背下每一行，而是之後自己開 agent 專案時，看到任何一段程式都能說出
「喔，這一部分是在幹麻」—— 迴圈、工具、記憶、提示、規劃、保護措施各自長什麼樣。

## Success looks like
- 打開 `execute_task`，能指出「問模型 → 有沒有 tool_calls → 執行工具 → 塞回歷史」四個區塊
- 能說明工具契約（JSON Schema）和工具實作（dispatch）為什麼是兩個地方、怎麼對上
- 能解釋 `conversation_history` 就是 agent 的記憶，以及多輪任務為什麼會記得前文
- 能看懂 system hint（時間戳、系統狀態、工具計數、TODO）是在每一輪「臨時加上去」而不是存進歷史
- 能說出 `max_iterations`、timeout、結果截斷這些保護措施各自防什麼
- 能打開 `trajectory.json` 回溯 agent 做過什麼

## Constraints
- 學習者是讀書會成員（會 Python、用過一次 LLM API 即可），facilitator 已熟悉檔案
- 每課 bite-size：10–15 分鐘讀完，一課只講一個面向
- 聚焦「面向 ↔ 對應程式碼」，不拘泥瑣碎細節；每課明列「可以先跳過」的段落
- 不需要實作單元（不留 TODO 給學員填）；回饋用「預測行為」小測驗
- 教材為 self-contained HTML，在瀏覽器開，搭配 VSCode 看原始碼
- 語言：繁體中文；程式註解與識別字保持原樣（英文）

## Out of scope
- 多 agent 協作、MCP、streaming、async
- Prompt engineering 技巧本身
- 模型訓練 / 後訓練
- 生產部署（sandbox、權限系統）—— 只點出 agent.py 目前缺什麼，不實作
