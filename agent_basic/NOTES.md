# Teaching Notes — agent_basic

## 使用者偏好
- 母語繁體中文；教材用中文，程式識別字保持英文。
- 本 track 用 **HTML lesson**（teach skill 預設）。根目錄 SVM track 的「一律 ipynb」偏好是 SVM 專屬（需要可跑的圖），這裡是導讀原始碼，不適用。使用者若要改成 ipynb 或 md 再換。
- **不要實作單元**（使用者明講）。回饋用預測行為的小測驗，不留程式填空。
- 不要拘泥瑣碎細節：每課放一個「可以先跳過」框。
- 不放「讀書會現場怎麼帶」之類的 facilitator 提示框（使用者要求刪除，2026-10-19）。
- 讀書會形式：使用者是 facilitator，開 VSCode 帶大家看；lesson 要能「邊看 lesson 邊在 VSCode 找到對應位置」。
- 引用程式位置：**函式名優先，行號為輔**（檔案還會改，行號會漂）。

## 背景（ZPD 起點）
- 使用者自己已讀過整份 agent.py、改過它（pydantic 化、log 清理、.env 模型切換），並寫了 `01_loop.py` 當最小版本。
- 學員假設：會 Python，呼叫過一次 chat API；可能沒看過 tool calling。

## 教學計畫（草案，依反應調整）
1. ✅ **0001 迴圈**：01_loop.py 的四個區塊 ↔ `execute_task`
2. ✅ 0002 工具：契約（`_get_tools_description` JSON Schema）vs 實作（`_execute_tool` dispatch + `_tool_*`）；兩處要對上；提 pydantic `model_json_schema()` 可以合一
3. ✅ 0003 記憶：`conversation_history` 是 list；system prompt；跨 `execute_task` 的多輪記憶；`reset()`
4. ✅ 0004 System hint：時間戳 / 系統狀態 / 工具計數 / 詳細錯誤；`messages_to_send = history.copy()` 的「臨時加上、不存檔」
5. ✅ 0005 規劃：TODO 工具 (`rewrite_todo_list`, `update_todo_status`)、`TodoItem`、system prompt 裡的規則、hint 注入 TODO
6. ✅ 0006 保護與觀測：`max_iterations`、timeout=30、結果截斷（agent.py 缺）、`shell=True` 風險、`FINAL ANSWER:` 契約、`trajectory.json`

## 工作流
- lesson 存 `lessons/NNNN-*.html`，共用 `assets/style.css` 與 `assets/quiz.js`。
- 行號來源：`grep -n "def " agent.py`；寫 lesson 前重跑一次。
