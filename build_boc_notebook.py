"""Builder for dinov2_boc_source_reading.ipynb (DINO-BoC official source walkthrough).
Single source of truth: edit section functions here, run `uv run python build_boc_notebook.py`
to regenerate the .ipynb, then execute it with nbconvert."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

FORK = "/Users/adamw/work/dinov2"
cells = []

def md(text):
    cells.append(new_markdown_cell(text))

def code(text):
    cells.append(new_code_cell(text))

def src(path, lineno, body):
    """A read-only source excerpt cell, shown as a fenced block with a file:line header."""
    md(f"**`{path}:{lineno}`**\n\n```python\n{body}\n```")

# ---- shared bootstrap source injected into the first runnable cell ----
BOOTSTRAP = f'''import sys, warnings
sys.path.insert(0, "{FORK}")            # 引入官方 dinov2 fork（不安裝、不動環境）
warnings.filterwarnings("ignore", message=".*xFormers.*")  # 只壓 xFormers 缺失警告，保留 glyph 警告可見
import torch, numpy as np
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["PingFang HK", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
print("torch", torch.__version__, "| MPS", torch.backends.mps.is_available())'''


def sec0():
    md("""# DINO-BoC 官方原始碼導讀
## 跟著一次真實 forward 資料流，讀懂 dinov2 怎麼實作 Bag-of-Channels

本筆記本是 `channel_adaptive_dino.ipynb`（toy 重實作，教**原理**）的姊妹篇，
專門教**官方 dinov2 怎麼寫** DINO-BoC。我們**讀真實原始碼**為主軸，並在**真實 import 的類別**上
跑極小 toy forward（<10 張合成圖）看 shape 變化。

**讀法**：跟著一次真實的特徵抽取資料流走 —
③ 資料層（通道被拆獨立）→ ④ config/hub（模型怎麼被建出來）→ ① ViT reshape（主幹）→ ② linear concat（融合）。

**環境前提**
- 以 `sys.path` 引入使用者 fork：`/Users/adamw/work/dinov2`（不安裝、不下載權重）。
- 缺 xFormers 會自動 fallback 到 native attention（只有 warning，可正常跑）。
- toy forward 用 `vit_small`（D=384）**代跑**真實 ViT-L/16（D=1024）：reshape 機制一模一樣，只有 D 數字不同。""")
    code(BOOTSTRAP)

def sec1():
    md("""## §1 鳥瞰：一次真實特徵抽取的資料流

下圖把四塊原始碼依**資料流**串起來。後面每一節各放大一塊。""")
    code('''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off")
stages = [
    ("3 資料層\\nSEPARATE_CHANNELS", "hpafov.py:250", "每通道拆成\\n獨立單通道樣本"),
    ("4 config / hub", "backbones.py:154", "channel_adaptive=True\\nin_chans=1"),
    ("1 ViT reshape", "vision_transformer.py:310", "x.reshape(B*C,1,H,W)\\n通道推入 batch"),
    ("2 linear concat", "linear.py:247", "torch.cat 各通道 CLS\\n→ (B, C*D)"),
]
n = len(stages)
for i,(title,loc,desc) in enumerate(stages):
    x = i/(n) + 0.02
    ax.add_patch(plt.Rectangle((x,0.35),0.20,0.5, fill=True, facecolor="#eef3fb",
                               edgecolor="#3a6ea5", lw=1.6, transform=ax.transAxes))
    ax.text(x+0.10,0.74,title,ha="center",va="center",fontsize=10,weight="bold",transform=ax.transAxes)
    ax.text(x+0.10,0.58,desc,ha="center",va="center",fontsize=8,transform=ax.transAxes)
    ax.text(x+0.10,0.42,loc,ha="center",va="center",fontsize=7,color="#a33",transform=ax.transAxes)
    if i<n-1:
        ax.annotate("",(x+0.22,0.6),(x+0.20,0.6),xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->",lw=1.5,color="#555"))
ax.text(0.5,0.06,"資料流：通道先被拆獨立 → trunk 只吃單通道 → eval 層再 concat 回來",
        ha="center",fontsize=9,style="italic",transform=ax.transAxes)
plt.tight_layout(); plt.show()''')


def main():
    cells.clear()
    sec0()
    sec1()
    nb = new_notebook(cells=list(cells))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, "dinov2_boc_source_reading.ipynb")
    print(f"wrote dinov2_boc_source_reading.ipynb with {len(cells)} cells")

if __name__ == "__main__":
    main()
