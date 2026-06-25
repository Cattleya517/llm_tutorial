# Mission: Support Vector Machines (SVM)

## Why
補齊經典機器學習的核心地基，支撐正在進行的 ViT / DINO 深度學習學習路線。
特別是「margin（間隔）」這個概念 —— 它在 SVM 裡講得最乾淨，之後會在
contrastive / metric learning（含 DINO）反覆出現。目標是建立扎實直覺，
而不是只會呼叫 `sklearn`。

## Success looks like
- 能用幾何直覺講清楚 SVM 在最佳化「什麼」（最大間隔），以及 support vector 的角色
- 看懂 hard margin → soft margin (C) → kernel trick 的脈絡，知道每一步在解決什麼問題
- 能在 2D 資料上用 sklearn 訓練、視覺化決策邊界與間隔，並解讀 `C`、`gamma`、`kernel`
- 能說出 SVM 與深度學習線性分類器 / margin 概念的關聯

## Constraints
- 直覺優先，數學點到為止（需要時才展開 dual / KKT，不一開始堆公式）
- 教材一律用 Jupyter notebook（`.ipynb`），不要 HTML
- 每一課可在 ~15–25 分鐘完成，搭配可實際執行的程式
- 環境：uv 專案，已裝 numpy / matplotlib / scikit-learn / ipywidgets

## Out of scope（暫時）
- SVM regression (SVR)、structured SVM
- 完整最佳化理論（SMO 內部機制、收斂證明）—— 之後想深入再開
- 大規模 / 生產部署調優
