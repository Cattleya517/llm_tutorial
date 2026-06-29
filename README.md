# LLM / Vision Tutorials

一系列 LLM 與 Vision 相關的 hands-on tutorial notebooks，使用 `uv` 管理依賴。

## 環境

```bash
uv sync                   # 安裝所有依賴
uv run jupyter lab        # 開 Jupyter Lab
```

Python 3.11+，主要依賴於 `pyproject.toml`。

## Notebooks

| 檔案 | 主題 | 內容 |
|---|---|---|
| `llm_api.ipynb` | LLM API | OpenAI / 相容 API 呼叫的基本範例 |
| `RAG/rag_demo.ipynb` | Retrieval-Augmented Generation | 用 PDF 建 Chroma vector DB、LangChain pipeline 查詢 |
| `ViT/vit_mnist.ipynb` | Vision Transformer 原理 | 從零實作 ViT（patch embed、attention、pos encoding），跑 MNIST |
| `ViT/vit_mnist_demo.ipynb` | ViT MNIST demo | `ViT/vit_mnist.ipynb` 的簡化展示版本 |
| `DINO/dinov2_pet.ipynb` | DINOv2 + 寵物分類 | 用預訓練 DINOv2-with-registers 做 kNN / Linear Probe，視覺化 attention map |

## 目錄結構

```
.
├── llm_api.ipynb         # LLM API tutorial
├── ViT/                  # Vision Transformer 系列（MNIST、CIFAR-10）
├── DINO/                 # DINOv2 / DINO 系列
├── RAG/                  # RAG 教學（含 PDF 教材與已建好的 Chroma DB）
├── cnn/                  # CNN 系列（retinopathy 等）
├── lessons/              # SVM 教學課程（/teach workspace：MISSION/NOTES/RESOURCES + lessons）
├── data/                 # 資料集（gitignored；notebooks 會自動下載）
├── pyproject.toml        # uv 依賴定義
└── .python-version       # Python 版本（3.11）
```

## 注意

- `data/` 為 gitignored；第一次跑各 notebook 會自動下載資料集
- `*_executed.ipynb` 也被 gitignored——是含執行輸出的副本，重跑即可重建
- RAG 教學的 PDF 與已建好的 `nano_manual_chroma_db` 都在 `RAG/` 內部，notebook 用相對路徑載入
