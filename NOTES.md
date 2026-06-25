# Teaching Notes — SVM

## 使用者偏好
- 教材一律用 **Jupyter notebook (`.ipynb`)**，不要 HTML（`lessons/` 以 `.ipynb` 存）。
- **直覺優先**，數學點到為止；需要時才展開推導。
- 喜歡「可實際跑」的程式 + 視覺化（沿用既有 ViT/DINO notebook 的學習方式）。
- 母語中文（繁體）。markdown 用中文解說；**matplotlib 圖內文字一律用英文**（環境無 CJK 字型，中文會變豆腐方塊）。

## 背景（ZPD 起點）
- 正在走 ViT → DINO 深度學習路線；熟 numpy、Python、基本線性代數（點積）。
- SVM 視為「補基礎」，不是全新領域 → 可大膽用幾何 / 向量語言，不必從零解釋向量。

## 教學計畫（草案，依反應調整）
1. ✅ **L1 最大間隔 + support vectors**（最寬街道直覺）
2. L2 Soft margin 與 `C`：資料不可分 / 有雜訊時怎麼辦（hinge loss 直覺）
3. L3 Kernel trick：非線性邊界，RBF 與 `gamma`
4. L4 多分類 + 實務調參（GridSearch），與 deep learning 線性探針的關聯

## 工作流
- 用 `build_lesson_NNNN.py`（nbformat）產生 notebook；`SOLVED=1 python build_lesson_NNNN.py` 可產生已填答版做執行驗證，預設產生留 TODO 的學習版。
