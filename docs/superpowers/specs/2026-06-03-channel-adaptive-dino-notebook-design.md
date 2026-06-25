# 設計文件：Channel-Adaptive DINO 論文講解 Notebook

- **日期**：2026-06-03
- **產出檔案**：`channel_adaptive_dino.ipynb`
- **論文**：De Lorenci et al., *Scaling Channel-Adaptive Self-Supervised Learning* (TMLR 06/2025) — 即 DINO-BoC 論文
- **風格參考**：`vit_cifar10.ipynb`（從零實作 + 圖解 + shape trace）、`dinov2_pet.ipynb`（DINO/SSL 概念 + 看出 forward pass）

## 目標與定位

學習軌跡第三站：把已學的 ViT / DINOv2 推廣到「通道異質（channel-heterogeneous）」的科學影像。
全程繁體中文、合成玩具資料、可完整重跑、**無需下載、無需訓練**。

核心敘事（論文 punchline）：在大規模下，**獨立編碼通道（DINO-BoC，late fusion）打敗聯合編碼
（Channel-ViT，early fusion）**，與「跨通道推理應該有用」的直覺相反。

## 關鍵決策（已與使用者確認）

1. 程式路線：**合成玩具資料 + 可完整前向跑**（非真實資料下載、非僅示意圖）。
2. 涵蓋架構：**三種全做** — Channel-ViT、DINO-BoC、DINO-HA。
3. 不做訓練：三個架構只做前向、看 shape / attention mask / 通道組合行為；數字用論文 bar chart 與
   「未見通道組合」機制示範來說明，**不以小規模訓練重現論文趨勢**（避免誤導）。

## 章節大綱

0. 標題 + 導讀（論文資訊、學習目標、punchline 預告、銜接前兩本）
1. 問題：通道異質性（數量/語意都不同）— 圖解重現 Fig 2 概念
2. 三種策略總覽 — 重現 Fig 3：Channel-Adaptive(early) ↔ HA(middle) ↔ BoC(late) 資料流圖
3. 合成多通道「細胞」資料 — 生成 K 通道玩具影像（核/膜/點狀…各通道不同結構），可重現、視覺化
4. 共同基礎：多通道 patchify — (K,H,W) → 單通道 patch 序列（§3.1 公式）+ shape trace + 圖解
5. 策略 A：Channel-ViT（聯合 / early fusion）— patch+pos+**channel embedding**、全跨通道 attention、前向、看 attention 矩陣；示範 K>K_max 結構性失效
6. 策略 B：DINO-BoC（獨立 / late fusion）— 單通道取樣→共享 backbone→逐通道特徵→concat；任意通道組合
7. 策略 C：DINO-HA（階層注意力）— global-CLS + 每通道-CLS、實作 attention mask、heatmap 圖解（重現 Fig 4）
8. 三者並排對比 — 三張 attention mask 並排 + 對比表（early/mid/late、可否任意 K、channel-agnostic?）
9. 論文主結果解讀 — bar chart 重現 headline（Table 1/4/8：BoC > Channel-ViT）+ 為何 late fusion 贏
10. 機制小實驗（看出來）— BoC 吃未見通道組合 OK、Channel-ViT 當 K>K_max 就壞；機械式看出泛化結論
11. 總結 + 延伸 — 何時用哪個、官方 channel-adaptive DINOv2 連結、學習軌跡銜接

## 技術選擇

- 依賴：`torch`、`numpy`、`matplotlib`、`sklearn`（皆與既有 notebook 一致，不新增重依賴）。
- 三個架構為小型可前向跑的真實實作；重點機制：patchify / channel-embedding / attention-mask。
- 透過 `uv run` 環境執行驗證（.venv 已具備所有依賴）。

## 成功標準

- 全 notebook 可從頭到尾 `uv run jupyter nbconvert --execute` 無錯誤跑完。
- 三個架構各自前向跑通並印出 shape；HA mask 與三者對比 mask 都能畫出。
- 風格、語氣、視覺化密度與兩本參考 notebook 一致。
