# 設計文件：DINO-BoC 官方原始碼導讀 Notebook

- **日期**：2026-06-04
- **產出檔案**：`dinov2_boc_source_reading.ipynb`（位於 `/Users/adamw/tutorials/llm_tutorial`）
- **來源庫**：使用者 fork 的官方 dinov2（`/Users/adamw/work/dinov2`，現已與 `facebookresearch/dinov2` main 同步，cell_dino 程式碼已 upstream）
- **論文**：De Lorenci et al., *Scaling Channel-Adaptive Self-Supervised Learning* (TMLR 06/2025) — 即 DINO-BoC
- **風格參考**：`channel_adaptive_dino.ipynb`（toy 重實作）、`vit_cifar10.ipynb`、`dinov2_pet.ipynb`

## 目標與定位

學習軌跡第四站（與第三站 `channel_adaptive_dino.ipynb` 互補）。

- 第三站（toy 筆記本）教 **原理**：from-scratch 重實作三種策略，看 shape / attention mask / 通道組合行為。
- **本筆記本教「官方 dinov2 怎麼寫」**：以官方真實原始碼為主軸，逐段導讀 BoC 的關鍵程式碼，並在**真實類別**上跑極小 toy forward 看 shape 變化。

全程繁體中文 bilingual，沿用既有筆記本風格與「0 glyph warning」標準。

## 關鍵決策（已與使用者 grill 確認）

1. **定位**：純原始碼導讀（讀官方真實碼），是 toy 筆記本的姊妹篇，不是重做 toy。
2. **執行邊界**：跑「低於十張 toy image」的 forward pass，**聚焦 shape 的變化**。即在真實 import 的 BoC 類別上跑極小合成 batch；**不下載任何權重**。
3. **原始碼範圍（四塊全覆蓋）**：
   - ① ViT reshape（核心）— `vision_transformer.py`
   - ② linear.py concat（核心）— `eval/cell_dino/linear.py`
   - ③ 資料層 SEPARATE_CHANNELS — `data/datasets/cell_dino/hpafov.py`
   - ④ config + hub 入口 — `configs/train/cell_dino/vitl16_boc_hpafov.yaml`、`hub/cell_dino/backbones.py`
4. **敘事順序**：跟著一次真實 forward 資料流 → ③資料層 → ④config/hub → ①ViT reshape → ②linear concat。
5. **原始碼呈現**：重點 highlight + 導讀；boilerplate（chunked_blocks、register tokens、各 branch）用 `...` 略過，不逐一展開。關鍵片段帶 `file:line` 標頭。
6. **模型大小**：toy forward 用 `vit_small`（D=384）快跑，明確標註「真實 BoC 是 ViT-L/16, D=1024」。

## 環境與技術前提（已驗證）

- 以 `sys.path.insert(0, "/Users/adamw/work/dinov2")` 讓 dinov2 可 import，**不安裝、不動環境**。
- 真實 `DinoVisionTransformer` 在 tutorials venv（torch 2.10, Mac MPS）可 import 並前向；缺 xformers 會自動 fallback 到 native attention（只有 warning，可正常跑）。
- 透過 `uv run` 執行驗證。
- 已驗證真實 BoC 路徑：`vit_small(patch_size=16, in_chans=1, channel_adaptive=True)` + `get_intermediate_layers(x, return_class_token=True)` 對 `x=(2,5,224,224)` 回傳 patch `(2,5,196,384)`、cls `(2,1920=5*384)`。
- 已驗證教學用 error：`forward_features` 對 5 通道輸入丟 `RuntimeError: expected input to have 1 channels, but got 5`，證明 trunk 為純單通道、late-fusion 縫在 `get_intermediate_layers` 進出口。

## 章節大綱

| § | 內容 | 對應原始碼 |
|---|------|-----------|
| 0 | 導讀：定位、與 toy 筆記本銜接、環境說明（`sys.path` 指向 fork、xformers fallback、用 vit_small 代跑 ViT-L、不下載權重） | — |
| 1 | 鳥瞰圖：一張 matplotlib 資料流圖把四塊串起，標出各自 `file:line` | 全部 |
| 2 | **③資料層** SEPARATE_CHANNELS：wildcard 把每通道拆成獨立單通道樣本，預訓練一次只看一個通道 → channel-agnostic 的源頭 | `data/datasets/cell_dino/hpafov.py` |
| 3 | **④模型怎麼被建出來**：config `channel_adaptive:true, in_chans:1` + hub `channel_adaptive_dino_vitl16` → `vision_transformer.py:107 self.bag_of_channels = channel_adaptive` | `configs/.../vitl16_boc_hpafov.yaml`、`hub/cell_dino/backbones.py`、`vision_transformer.py:107` |
| 4 | **①ViT reshape（核心）**：`get_intermediate_layers` 進口 `x.reshape(B*C,1,H,W)`、出口把 cls reshape 回 `(B, C*D)`；**live toy forward**（vit_small, B=2, C=5）印 `(2,5,224,224)→patch(2,5,196,384)、cls(2,1920)`；示範 `forward_features` 對 5 通道丟 error 的教學意義 | `vision_transformer.py:310/329` |
| 5 | **②linear concat（核心）**：`create_linear_input` 的 late fusion = `torch.cat` per-channel CLS（+optional avgpool of patch tokens），手動重現 `(B, C*D)` | `eval/cell_dino/linear.py:247` |
| 6 | toy ↔ 真實原始碼對照表（接 `channel_adaptive_dino.ipynb`）、HA 不在官方碼的提醒、延伸閱讀 | — |

## 核心敘事（因果鏈）

§2→§5 是一條完整因果鏈：**資料層先把通道拆獨立 → trunk 因此只會吃單通道 → 所以 eval 層必須用 `torch.cat` 把各通道 CLS 重新拼起來**。讀者讀完會理解「為什麼非得 late fusion 不可」是被資料層設計逼出來的，而非任意選擇。

§4 的「error 當教材」是點睛：`forward_features(5-channel)` 報錯直接證明 trunk 是純單通道；BoC 的「架構」其實只是 `get_intermediate_layers` 進出口各一行 reshape 約定，沒有新模組、沒有 channel embedding。

## 技術選擇

- 依賴：`torch`、`numpy`、`matplotlib`（與既有 notebook 一致，不新增重依賴）；外加 `sys.path` 指向 dinov2 fork。
- 真實原始碼片段以程式碼 cell 或 markdown 顯示（帶 `file:line` 標頭），導讀文字為繁中 markdown。
- toy forward 用 `vit_small` 隨機初始化，CPU/MPS 幾秒內完成。

## 成功標準

- 全 notebook 可從頭到尾 `uv run jupyter nbconvert --execute` 無錯誤跑完（含 §4 的「預期 error」用 try/except 包起來示範，不讓 nbconvert 中斷）。
- §4 live forward 印出與本文件一致的 shape；§5 手動 concat 印出 `(B, C*D)`。
- §1 資料流圖能畫出且無 glyph warning。
- 四塊原始碼各自有 `file:line` 標頭與導讀；對照表接得上 `channel_adaptive_dino.ipynb`。
- 風格、語氣、視覺化密度與既有三本 notebook 一致。

## 明確不做（YAGNI）

- 不下載/載入任何預訓練權重（gated、多 GB、需 CUDA）。
- 不跑訓練、不重現論文數字。
- 不導讀 HA（官方碼不含 HA，已在 toy 筆記本以論文 §3.3 重建）。
- 不安裝 dinov2 套件、不修改 venv（只用 `sys.path`）。
