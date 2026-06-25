# DINO-BoC 官方原始碼導讀 Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 產出 `dinov2_boc_source_reading.ipynb` — 一本繁中原始碼導讀筆記本,跟著一次真實 forward 資料流,逐段導讀官方 dinov2 的 DINO-BoC 四塊關鍵程式碼,並在真實類別上跑極小 toy forward 看 shape。

**Architecture:** 用一支 `nbformat` builder 腳本（`build_boc_notebook.py`）作為唯一真實來源,逐 task 加入各章節的 cell;每個 task 重新產生 `.ipynb`、用 `uv run jupyter nbconvert --execute` 跑完、grep 輸出驗證 shape/字串,再 commit。dinov2 fork 以 `sys.path.insert` 引入,不安裝、不下載權重;toy forward 用 `vit_small`(D=384) 代跑真實 ViT-L。

**Tech Stack:** Python 3.11, `nbformat`, `torch` 2.10 (MPS), `numpy`, `matplotlib`(PingFang HK CJK 字型), `uv run`, `jupyter nbconvert`。

---

## File Structure

- **Create**: `build_boc_notebook.py` — nbformat builder,唯一真實來源。內含 helper(`md()`, `code()`, `src()` 帶 `file:line` 標頭)、CJK 字型設定字串、各章節 builder 函式 `sec0()`…`sec6()`、`main()` 組裝並寫出 `.ipynb`。
- **Generate**: `dinov2_boc_source_reading.ipynb` — builder 產出、`nbconvert --execute --inplace` 跑過(輸出與圖內嵌)。
- **Read-only 參考(絕不修改)**:
  - `/Users/adamw/work/dinov2/dinov2/models/vision_transformer.py`（`:107` bag_of_channels、`:305-345` get_intermediate_layers reshape）
  - `/Users/adamw/work/dinov2/dinov2/eval/cell_dino/linear.py`（`:247` create_linear_input）
  - `/Users/adamw/work/dinov2/dinov2/data/datasets/cell_dino/hpafov.py`（`:107`、`:234-235`、`:250-258`、`:270-272`）
  - `/Users/adamw/work/dinov2/dinov2/configs/train/cell_dino/vitl16_boc_hpafov.yaml`
  - `/Users/adamw/work/dinov2/dinov2/hub/cell_dino/backbones.py`（`:154` channel_adaptive_dino_vitl16）

## 慣例（所有 task 共用）

- 全程 `uv run`（tutorials venv 有 torch 2.10）。
- 從專案目錄 `/Users/adamw/tutorials/llm_tutorial` 執行(否則 uv 解析不到本專案 venv)。
- 繁中 bilingual、heavy shape trace、沿用既有 notebook 風格。
- **0 glyph warning 標準**:matplotlib 圖內文字**不可**用 PingFang HK 缺的字元 — 禁用 `✓`(U+2713)、`⚠`(U+26A0)、`✗`(U+2717);要強調改用「注意：」「OK」「FAIL」等純文字或 ASCII。
- nbconvert 用隔離 jupyter config 避免讀到壞掉的全域 config(見 Task 0 指令)。

---

### Task 0: Builder 骨架 + §0 導讀 + bootstrap cell

**Files:**
- Create: `build_boc_notebook.py`
- Generate: `dinov2_boc_source_reading.ipynb`

- [ ] **Step 1: 建立 builder 骨架與 helper**

建立 `build_boc_notebook.py`:

```python
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
```

- [ ] **Step 2: 寫 §0 導讀 + main()**

接在 `build_boc_notebook.py` 後面加 `sec0()` 與 `main()`:

```python
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

def main():
    cells.clear()
    sec0()
    nb = new_notebook(cells=list(cells))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, "dinov2_boc_source_reading.ipynb")
    print(f"wrote dinov2_boc_source_reading.ipynb with {len(cells)} cells")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 產生 notebook**

Run: `cd /Users/adamw/tutorials/llm_tutorial && uv run python build_boc_notebook.py`
Expected: `wrote dinov2_boc_source_reading.ipynb with 3 cells`

- [ ] **Step 4: 執行 notebook 驗證 bootstrap 跑得起來**

Run:
```bash
cd /Users/adamw/tutorials/llm_tutorial && \
JUPYTER_CONFIG_DIR=/tmp/jcfg uv run jupyter nbconvert --to notebook --execute --inplace \
  dinov2_boc_source_reading.ipynb 2>&1 | tail -3
```
Expected: 無 error;結尾顯示 `Writing ... dinov2_boc_source_reading.ipynb`。

- [ ] **Step 5: 確認 torch/MPS 那行有印出來**

Run: `cd /Users/adamw/tutorials/llm_tutorial && uv run python -c "import json; nb=json.load(open('dinov2_boc_source_reading.ipynb')); print([o for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='stream'])"`
Expected: 看到含 `torch 2.10` 與 `MPS True` 的輸出。

- [ ] **Step 6: Commit**

```bash
cd /Users/adamw/tutorials/llm_tutorial
git add build_boc_notebook.py dinov2_boc_source_reading.ipynb
git commit -m "feat(boc-nb): scaffold builder + §0 導讀 + bootstrap"
```

---

### Task 1: §1 鳥瞰資料流圖

**Files:**
- Modify: `build_boc_notebook.py`(新增 `sec1()`,並在 `main()` 的 `sec0()` 後呼叫)

- [ ] **Step 1: 新增 §1 — 一張把四塊串起來的 matplotlib 資料流圖**

在 builder 加入(放在 `sec0` 之後):

```python
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
```

並在 `main()` 內 `sec0()` 下一行加 `sec1()`。

- [ ] **Step 2: 重新產生並執行**

Run:
```bash
cd /Users/adamw/tutorials/llm_tutorial && uv run python build_boc_notebook.py && \
JUPYTER_CONFIG_DIR=/tmp/jcfg uv run jupyter nbconvert --to notebook --execute --inplace \
  dinov2_boc_source_reading.ipynb 2>&1 | grep -iE "glyph|error|warning" | grep -vi "xformers" | head
```
Expected: **無任何 glyph/error 輸出**(空)。

- [ ] **Step 3: 確認圖有內嵌**

Run: `cd /Users/adamw/tutorials/llm_tutorial && uv run python -c "import json; nb=json.load(open('dinov2_boc_source_reading.ipynb')); print('has_png', any('image/png' in o.get('data',{}) for c in nb['cells'] for o in c.get('outputs',[])))"`
Expected: `has_png True`

- [ ] **Step 4: Commit**

```bash
cd /Users/adamw/tutorials/llm_tutorial
git add build_boc_notebook.py dinov2_boc_source_reading.ipynb
git commit -m "feat(boc-nb): §1 資料流鳥瞰圖"
```

---

### Task 2: §2 資料層 SEPARATE_CHANNELS

**Files:**
- Modify: `build_boc_notebook.py`(新增 `sec2()` 並在 `main()` 呼叫)

- [ ] **Step 1: 新增 §2 — 導讀 + 真實原始碼片段 + numpy 拆通道 demo**

`hpafov.py` 的關鍵：`SEPARATE_CHANNELS` wildcard 把每個通道拆成獨立單通道樣本(下面第 250-258 行的 `np.concatenate` 攤平)。加入 builder:

```python
def sec2():
    md("""## §2 資料層：`SEPARATE_CHANNELS`（channel-agnostic 的源頭）

BoC 的「通道無關」不是模型決定的，是**資料層**決定的：預訓練時每個通道被當成一張**獨立的單通道樣本**，
backbone 一次只看一個通道，所以它根本沒機會學「跨通道」的東西。""")
    src("dinov2/data/datasets/cell_dino/hpafov.py", "107",
        'SEPARATECHANNELS = "separate_channels"  # each channel from each image\\n'
        '# is treated as an independent sample, overrides chosen channel configuration')
    src("dinov2/data/datasets/cell_dino/hpafov.py", "250-258",
        '# 原本每張影像有 4 個通道索引：shape (num_images, 4)\\n'
        'self._channels = np.repeat(np.array([[0, 1, 2, 3]]), len(self._image_paths), axis=0).tolist()\\n'
        '...\\n'
        'if self.wildcard == _WildCard.SEPARATECHANNELS.value.upper():\\n'
        '    C = channels.shape[1]\\n'
        '    # 把通道維「攤平」進樣本維：4 通道 x N 張 → 4N 個單通道樣本\\n'
        '    channels = np.concatenate([channels[:, i] for i in range(C)])\\n'
        '    self._channels = np.expand_dims(channels, 1).tolist()')
    md("用一個迷你 numpy demo 看這個「攤平」到底做了什麼（拿 3 張 4 通道影像示意）：")
    code('''N, C = 3, 4
channels = np.repeat(np.array([[0,1,2,3]]), N, axis=0)
print("拆之前 channels.shape :", channels.shape, "  (N 張, 每張 4 通道)")
flat = np.concatenate([channels[:, i] for i in range(C)])
sep = np.expand_dims(flat, 1)
print("拆之後 sep.shape      :", sep.shape, "  (4N 個獨立單通道樣本)")
print("每個樣本只帶 1 個通道索引：", sep[:,0].tolist())''')
    md("""> 結論：trunk 之所以是**純單通道**(`in_chans=1`)，是因為資料層在更上游就保證了「一次只餵一個通道」。
> 這條設計把後面 §4/§5 的 late-fusion **逼了出來**——融合只能放在後面做。""")
```

並在 `main()` 加 `sec2()`。

- [ ] **Step 2: 重新產生並執行,驗證 demo 輸出**

Run:
```bash
cd /Users/adamw/tutorials/llm_tutorial && uv run python build_boc_notebook.py && \
JUPYTER_CONFIG_DIR=/tmp/jcfg uv run jupyter nbconvert --to notebook --execute --inplace \
  dinov2_boc_source_reading.ipynb 2>&1 | grep -iE "error" | grep -vi xformers | head
```
Expected: 無 error 輸出。

- [ ] **Step 3: 確認攤平 demo 印出 (3,4)→(12,1)**

Run: `cd /Users/adamw/tutorials/llm_tutorial && uv run python -c "import json; nb=json.load(open('dinov2_boc_source_reading.ipynb')); t=''.join(o.get('text','') for c in nb['cells'] for o in c.get('outputs',[])); print('(12, 1)' in t and '(3, 4)' in t)"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
cd /Users/adamw/tutorials/llm_tutorial
git add build_boc_notebook.py dinov2_boc_source_reading.ipynb
git commit -m "feat(boc-nb): §2 資料層 SEPARATE_CHANNELS 導讀 + 拆通道 demo"
```

---

### Task 3: §3 config / hub — 模型怎麼被建出來

**Files:**
- Modify: `build_boc_notebook.py`(新增 `sec3()` 並在 `main()` 呼叫)

- [ ] **Step 1: 新增 §3 — config + hub 原始碼 + 建出真實模型驗證旗標**

加入 builder:

```python
def sec3():
    md("""## §3 模型怎麼被建出來：config → hub → `self.bag_of_channels`

BoC **不需要新模型**：就是一個標準 DINOv2 ViT，差別只在兩個旗標 `channel_adaptive: true` 與 `in_chans: 1`。
它們從 train config / hub entrypoint 一路傳到 `DinoVisionTransformer.__init__`。""")
    src("dinov2/configs/train/cell_dino/vitl16_boc_hpafov.yaml", "4-9",
        'train:\\n  channel_adaptive: true        # 開啟 BoC\\n'
        'student:\\n  arch: vit_large\\n  patch_size: 16\\n  in_chans: 1   # 單通道 backbone')
    src("dinov2/hub/cell_dino/backbones.py", "154-180",
        'def channel_adaptive_dino_vitl16(\\n'
        '    ...\\n    channel_adaptive: bool = True,\\n'
        '):\\n    ...\\n    return _make_cell_dino_model(\\n'
        '        ..., in_chans=in_channels, channel_adaptive=channel_adaptive)')
    src("dinov2/models/vision_transformer.py", "107",
        'self.bag_of_channels = channel_adaptive   # 旗標落地：之後所有 reshape 都看它')
    md("我們在**真實類別**上把這條鏈走一遍（用 vit_small 代跑 ViT-L，旗標行為一致）：")
    code('''from dinov2.models.vision_transformer import vit_small
model = vit_small(patch_size=16, in_chans=1, channel_adaptive=True, img_size=224).eval()
print("型別           :", type(model).__name__)
print("in_chans (patch):", model.patch_embed.proj.in_channels, " <- 單通道")
print("bag_of_channels :", model.bag_of_channels, "        <- BoC 開啟")''')
```

並在 `main()` 加 `sec3()`。

- [ ] **Step 2: 重新產生並執行**

Run:
```bash
cd /Users/adamw/tutorials/llm_tutorial && uv run python build_boc_notebook.py && \
JUPYTER_CONFIG_DIR=/tmp/jcfg uv run jupyter nbconvert --to notebook --execute --inplace \
  dinov2_boc_source_reading.ipynb 2>&1 | grep -iE "error|traceback" | grep -vi xformers | head
```
Expected: 無 error。

- [ ] **Step 3: 驗證旗標輸出**

Run: `cd /Users/adamw/tutorials/llm_tutorial && uv run python -c "s=open('dinov2_boc_source_reading.ipynb').read(); print('bag_of_channels : True' in s and 'in_chans (patch): 1' in s)"`
Expected: `True`
(注意:nbformat 的 output `text` 是 list of strings,直接讀原始 JSON 檔做 substring 比 `''.join(o.get('text',''))` 穩。)

- [ ] **Step 4: Commit**

```bash
cd /Users/adamw/tutorials/llm_tutorial
git add build_boc_notebook.py dinov2_boc_source_reading.ipynb
git commit -m "feat(boc-nb): §3 config/hub → bag_of_channels 旗標落地"
```

---

### Task 4: §4 ViT reshape（核心）+ error 當教材

**Files:**
- Modify: `build_boc_notebook.py`(新增 `sec4()` 並在 `main()` 呼叫)

- [ ] **Step 1: 新增 §4 — reshape 原始碼 + live toy forward + 預期 error demo**

加入 builder(`get_intermediate_layers` 進口 reshape + live forward + try/except 的 error 示範):

```python
def sec4():
    md("""## §4 核心①：ViT reshape —— BoC 的靈魂只有兩行

BoC 不在主幹裡。融合縫在 `get_intermediate_layers` 的**進出口各一行 reshape**：
進口把通道推進 batch 維（讓單通道 backbone 逐通道各跑一次），出口再把 CLS 攤回 `(B, C*D)`。""")
    src("dinov2/models/vision_transformer.py", "310",
        'if self.bag_of_channels:\\n'
        '    B, C, H, W = x.shape\\n'
        '    x = x.reshape(B * C, 1, H, W)  # 進口：通道推入 batch，逐通道各跑一次單通道 ViT')
    src("dinov2/models/vision_transformer.py", "329-345",
        'if self.bag_of_channels:\\n'
        '    ...\\n'
        '    # 出口：patch token 還原成 (B, C, N, D)、class token 攤平成 (B, C*D)\\n'
        '    patch_tokens = [pt.reshape(B, C, pt.shape[-2], pt.shape[-1]) for pt in patch_tokens_per_block]\\n'
        '    cls_tokens   = [ct.reshape(B, -1) for ct in cls_tokens_per_block]')
    md("**Live toy forward**（B=2 張、C=5 通道、224×224；vit_small 隨機初始化）。注意 cls 出來是 `(B, C*D)`：")
    code('''x = torch.randn(2, 5, 224, 224)   # B=2 張，C=5 通道（<10 張，聚焦 shape）
with torch.no_grad():
    outs = model.get_intermediate_layers(x, n=1, return_class_token=True, reshape=False)
patch, cls = outs[0]
print("輸入            :", tuple(x.shape))
print("patch tokens    :", tuple(patch.shape), "  = (B, C, N, D)，N=196 個 patch")
print("class tokens    :", tuple(cls.shape),   "      = (B, C*D) = (2, 5*384)")
assert tuple(patch.shape) == (2, 5, 196, 384)
assert tuple(cls.shape) == (2, 1920)''')
    md("""**error 當教材**：直接餵 5 通道給 `forward_features`（訓練主幹入口）會炸——
因為 patch_embed 是 `in_chans=1`。這正證明 trunk 是**純單通道**、late-fusion 不在主幹裡：""")
    code('''try:
    model.forward_features(torch.randn(2, 5, 224, 224))
except RuntimeError as e:
    print("如預期報錯：", str(e).split(chr(10))[0])''')
```

並在 `main()` 加 `sec4()`。

- [ ] **Step 2: 重新產生並執行(assert 必須通過)**

Run:
```bash
cd /Users/adamw/tutorials/llm_tutorial && uv run python build_boc_notebook.py && \
JUPYTER_CONFIG_DIR=/tmp/jcfg uv run jupyter nbconvert --to notebook --execute --inplace \
  dinov2_boc_source_reading.ipynb 2>&1 | grep -iE "error|traceback|assert" | grep -vi xformers | head
```
Expected: 無 error（assert 通過代表 shape 正確；若 shape 不符會在此中斷）。

- [ ] **Step 3: 驗證 shape 與 error 字串都印出來**

Run: `cd /Users/adamw/tutorials/llm_tutorial && uv run python -c "s=open('dinov2_boc_source_reading.ipynb').read(); print('(2, 1920)' in s and '(2, 5, 196, 384)' in s and 'got 5 channels' in s)"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
cd /Users/adamw/tutorials/llm_tutorial
git add build_boc_notebook.py dinov2_boc_source_reading.ipynb
git commit -m "feat(boc-nb): §4 ViT reshape 核心 + live forward + error 教材"
```

---

### Task 5: §5 linear concat（核心）

**Files:**
- Modify: `build_boc_notebook.py`(新增 `sec5()` 並在 `main()` 呼叫)

- [ ] **Step 1: 新增 §5 — create_linear_input 原始碼 + 手動重現 concat**

加入 builder:

```python
def sec5():
    md("""## §5 核心②：`create_linear_input` —— late fusion 就是一個 `torch.cat`

eval 層怎麼把逐通道的 CLS 變成一條影像特徵？答案就是把各通道 class token `torch.cat` 起來
（BoC 時 class token 已是 `(B, C*D)`），需要時再接上 patch token 的 avgpool。""")
    src("dinov2/eval/cell_dino/linear.py", "247",
        'def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool, bag_of_channels):\\n'
        '    intermediate_output = x_tokens_list[-use_n_blocks:]\\n'
        '    # late fusion：把最後幾個 block 的 class token 串接\\n'
        '    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)\\n'
        '    if bag_of_channels and use_avgpool:\\n'
        '        # 可選：再接上 patch token 對 N 取平均（各通道）\\n'
        '        output = torch.cat((output, torch.mean(intermediate_output[-1][0], dim=-2)\\n'
        '                            .reshape(intermediate_output[-1][0].shape[0], -1)), dim=-1)\\n'
        '    return output.reshape(output.shape[0], -1).float()')
    md("用 §4 跑出來的 `(patch, cls)` 手動重現這個融合（取 1 個 block、不接 avgpool）：")
    code('''x_tokens_list = [(patch, cls)]          # 模擬 get_intermediate_layers 的回傳
inter = x_tokens_list[-1:]               # use_n_blocks=1
fused = torch.cat([ct for _, ct in inter], dim=-1)   # late fusion
print("融合後影像特徵 :", tuple(fused.shape), "  = (B, C*D)")
print("逐張獨立        :", "每張 5 個通道的 CLS 串成一條 1920 維向量")
assert tuple(fused.shape) == (2, 1920)''')
    md("""> 與直覺相反的 punchline：跨通道推理被刻意**拿掉**了（trunk 看不到別的通道），
> 融合只發生在最後這個 `cat`。論文證明在大規模下，這種 late fusion **打敗** early fusion 的 Channel-ViT。""")
```

並在 `main()` 加 `sec5()`。

- [ ] **Step 2: 重新產生並執行**

Run:
```bash
cd /Users/adamw/tutorials/llm_tutorial && uv run python build_boc_notebook.py && \
JUPYTER_CONFIG_DIR=/tmp/jcfg uv run jupyter nbconvert --to notebook --execute --inplace \
  dinov2_boc_source_reading.ipynb 2>&1 | grep -iE "error|traceback|assert" | grep -vi xformers | head
```
Expected: 無 error。

- [ ] **Step 3: 驗證融合 shape**

Run: `cd /Users/adamw/tutorials/llm_tutorial && uv run python -c "print(open('dinov2_boc_source_reading.ipynb').read().count('(2, 1920)') >= 2)"`
Expected: `True`(§4 與 §5 各出現一次)

- [ ] **Step 4: Commit**

```bash
cd /Users/adamw/tutorials/llm_tutorial
git add build_boc_notebook.py dinov2_boc_source_reading.ipynb
git commit -m "feat(boc-nb): §5 linear concat 核心 + 手動重現 late fusion"
```

---

### Task 6: §6 對照表 + 收尾 + 全本驗收

**Files:**
- Modify: `build_boc_notebook.py`(新增 `sec6()` 並在 `main()` 呼叫)

- [ ] **Step 1: 新增 §6 — toy↔真實對照表、HA 提醒、延伸**

加入 builder:

```python
def sec6():
    md("""## §6 對照表與收尾

### toy 筆記本 ↔ 官方原始碼對照

| 概念 | `channel_adaptive_dino.ipynb`（toy） | 官方 dinov2 原始碼 |
|------|--------------------------------------|--------------------|
| 通道拆獨立 | 手動逐通道切片 | `hpafov.py:250` `np.concatenate` 攤平 |
| 單通道 backbone | `BoCBackbone`（無 channel embed） | `vit_*(in_chans=1, channel_adaptive=True)` |
| 通道推入 batch | `for k in range(K)` 迴圈 | `vision_transformer.py:310` `x.reshape(B*C,1,H,W)` |
| late fusion | `torch.cat(feats, 0)` | `linear.py:247` `create_linear_input` |
| 旗標 | 常數 `K, D` | `config: channel_adaptive/in_chans` |

### 兩個提醒
- **官方碼只含 BoC，不含 HA（Hierarchical Attention）。** toy 筆記本的 HA 是依論文 §3.3 重建的，官方 repo 沒有對應檔。
- toy 用 `D=48`、官方 ViT-L 用 `D=1024`、本筆記本 demo 用 `vit_small D=384`：**reshape 機制完全相同**，只有維度數字不同。

### 延伸
- 真實權重推論需下載 gated checkpoint 並用 `torch.hub.load(..., 'channel_adaptive_dino_vitl16')`（見 fork 的 `notebooks/cell_dino/inference.ipynb`），需 CUDA 與多 GB 權重，本筆記本刻意不做。
- 回頭看 `channel_adaptive_dino.ipynb` 的三策略對比（Channel-ViT / BoC / HA），現在你能把每一塊對到真實原始碼了。""")
```

並在 `main()` 加 `sec6()`。

- [ ] **Step 2: 重新產生並全本執行(最終驗收)**

Run:
```bash
cd /Users/adamw/tutorials/llm_tutorial && uv run python build_boc_notebook.py && \
JUPYTER_CONFIG_DIR=/tmp/jcfg uv run jupyter nbconvert --to notebook --execute --inplace \
  dinov2_boc_source_reading.ipynb 2>&1 | tee /tmp/boc_run.log | tail -3
```
Expected: 結尾 `Writing ... dinov2_boc_source_reading.ipynb`,全程無 error。

- [ ] **Step 3: 驗收 — 0 glyph warning、0 error、圖有內嵌**

Run:
```bash
cd /Users/adamw/tutorials/llm_tutorial && \
echo "glyph/error 行數(應為0):" && grep -iE "glyph|^Error|Traceback" /tmp/boc_run.log | grep -vi xformers | wc -l && \
uv run python -c "import json; nb=json.load(open('dinov2_boc_source_reading.ipynb')); print('cells', len(nb['cells'])); print('has_png', any('image/png' in o.get('data',{}) for c in nb['cells'] for o in c.get('outputs',[])))"
```
Expected: glyph/error 行數 `0`;`cells` 約 30+;`has_png True`。

- [ ] **Step 4: Commit**

```bash
cd /Users/adamw/tutorials/llm_tutorial
git add build_boc_notebook.py dinov2_boc_source_reading.ipynb
git commit -m "feat(boc-nb): §6 對照表收尾 + 全本驗收通過"
```

---

## 完成定義

- `dinov2_boc_source_reading.ipynb` 從頭到尾 nbconvert 執行無 error、0 glyph warning。
- §4 印出 `patch (2,5,196,384)`、`cls (2,1920)`，error demo 印出 `got 5 channels`。
- §5 手動 concat 印出 `(2, 1920)`。
- §1 資料流圖內嵌。
- 四塊原始碼(`hpafov.py:250`、config/`backbones.py:154`、`vision_transformer.py:310`、`linear.py:247`)各有 `file:line` 標頭與導讀;§6 對照表接得上 `channel_adaptive_dino.ipynb`。
