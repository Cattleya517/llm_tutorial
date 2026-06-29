"""Build lesson 0001 — Maximum margin: the widest street.

預設產生「留 TODO 的學習版」。
執行 `SOLVED=1 .venv/bin/python build_lesson_0001.py` 會產生已填答版（供執行驗證用）。
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

SOLVED = bool(os.environ.get("SOLVED"))
OUT = "lessons/0001-maximum-margin-the-widest-street.ipynb"

cells = []
def md(src):  cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

# ── 1. 標題 + 框架 ────────────────────────────────────────────────
md(r"""# Lesson 1 — 最大間隔：找出「最寬的那條街」

> **SVM 學習路線 · 第 1 課** ｜ 預計 15–25 分鐘
> Mission：補 ML 基礎理論，作為 ViT / DINO 深度學習路線的地基。

當兩類資料可以被一條線分開時，**有無限多條線都能分對**。
SVM 憑什麼挑出「最好」的那一條？答案就是這一課唯一要教的東西：
**最大間隔 (maximum margin)**。

學完你會：
- 用幾何直覺說出 SVM 在最佳化「什麼」
- 在圖上認出 **support vectors（支持向量）**，並說明為什麼只有它們重要
- 自己寫出決策函數 `f(x)=w·x+b` 與間隔寬度 `2/‖w‖`，並與 sklearn 對照

> 💬 **我是你的老師**：notebook 裡任何看不懂的地方，直接回到對話問我。
> 想更深、想看 dual/KKT 數學推導、卡住了，都可以喊。
""")

# ── 2. 為什麼這對你的深度學習路線重要 ─────────────────────────────
md(r"""## 為什麼從「margin」開始？

你正在學 ViT / DINO。**「margin（間隔）」是一條會反覆出現的主線**：
- 深度學習的線性分類器（linear probe）也是在切一個超平面 `w·x+b`；
- contrastive / metric learning（DINO 的近親）想把同類拉近、異類推遠 —— 本質就是在**擴大 margin**。

SVM 是把「margin」這個概念講得**最乾淨**的地方：沒有反向傳播、沒有隨機性，
就是一個漂亮的幾何最佳化問題。先在這裡建立扎實直覺，回到深度學習會更通透。

> 📖 來源：[scikit-learn SVM User Guide](https://scikit-learn.org/stable/modules/svm.html) ·
> [Python Data Science Handbook — SVM (VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html)
""")

# ── 3. 載入套件 + 造資料 ─────────────────────────────────────────
code(r"""import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

plt.rcParams.update({"figure.figsize": (6, 6), "font.size": 12})

# 經典可分資料（取自 Python Data Science Handbook）
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=60, edgecolors="k")
plt.title("Two classes — how would YOU draw the boundary?")
plt.xlabel("x1"); plt.ylabel("x2")
plt.show()
""")

# ── 4. 問題：無限多條線 ──────────────────────────────────────────
md(r"""## 問題：哪一條線才「好」？

下面三條線**都把資料 100% 分對**了。它們一樣好嗎？
注意看：有的線貼著某個點擦身而過，有的則離兩邊都遠遠的。
""")

code(r"""xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=60, edgecolors="k")

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, "--k")

plt.xlim(-1, 3.5)
plt.title("Three lines, all separate perfectly — equally good?")
plt.xlabel("x1"); plt.ylabel("x2")
plt.show()
""")

# ── 5. SVM 的答案：最寬的街 ──────────────────────────────────────
md(r"""## SVM 的答案：畫一條「街」，挑最寬的

關鍵轉念：別把分界想成一條**零寬度的線**，而是想成一條有寬度的**街道 (street)**。
每條候選線都能往兩側加寬，直到**碰到最近的資料點**為止 —— 那就是它的街寬。

> **SVM 選的，就是那條能撐開最寬街道的線。**

直覺上為什麼？街越寬，分界離兩邊資料越遠，對雜訊與新資料越穩 →
**泛化誤差越低**（larger margin → lower generalization error，
[scikit-learn](https://scikit-learn.org/stable/modules/svm.html)）。

下圖把每條線的「街」畫成灰色帶，**並直接算出寬度** —— 街往兩側加寬，直到碰到最近的點為止：
""")

code(r"""plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=60, edgecolors="k")

# 街往兩側加寬，直到碰到最近的點為止；回傳 (鉛直半寬 v_off, 真正垂直街寬 width)
def street(m, b, X):
    resid = m * X[:, 0] - X[:, 1] + b          # 每個點到線的「帶號鉛直殘差」
    v_off = np.abs(resid).min()                # 撐到碰最近點所需的鉛直位移
    width = 2 * v_off / np.sqrt(m**2 + 1)      # 換算成垂直於線的真實街寬
    return v_off, width

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    v_off, width = street(m, b, X)
    yfit = m * xfit + b
    plt.plot(xfit, yfit, "-k")
    plt.fill_between(xfit, yfit - v_off, yfit + v_off,
                     edgecolor="none", color="#888", alpha=0.35)
    print(f"line  y = {m}·x + {b}   →   street width = {width:.3f}")

plt.xlim(-1, 3.5)
plt.title("Each line widened until it touches the nearest point")
plt.xlabel("x1"); plt.ylabel("x2")
plt.show()
""")

# ── 6. 讓 SVM 自己找最寬街 ───────────────────────────────────────
md(r"""## 讓 SVM 親自找出最寬街

現在交給 `sklearn`。我們用 `kernel="linear"` 的 `SVC`，並把 `C` 設超大
（≈ 不容許任何點越界，也就是 **hard margin**；`C` 是下一課主角，這裡先別管）。

下圖：
- **實線** = 決策邊界（`w·x+b = 0`）
- **虛線** = 街的兩個邊緣（`w·x+b = ±1`）
- **綠圈** = **support vectors**：剛好頂在街緣上的點
""")

code(r"""clf = SVC(kernel="linear", C=1e10)   # 超大 C ≈ hard margin
clf.fit(X, y)

def plot_svc_decision_function(model, ax=None):
    ax = ax or plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx = np.linspace(*xlim, 30)
    yy = np.linspace(*ylim, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    P = model.decision_function(xy).reshape(XX.shape)
    # 邊界(0) 與街緣(±1)
    ax.contour(XX, YY, P, levels=[-1, 0, 1], colors="k",
               linestyles=["--", "-", "--"], alpha=0.7)
    # 圈出 support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=260, facecolors="none", edgecolors="limegreen", linewidths=2.2)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=60, edgecolors="k")
plot_svc_decision_function(clf)
plt.title(f"SVM's widest street — {len(clf.support_vectors_)} support vectors (green)")
plt.xlabel("x1"); plt.ylabel("x2")
plt.show()

print("support vector 的索引:", clf.support_)
print("w =", clf.coef_[0], "  b =", clf.intercept_[0])
""")

# ── 7. 重點：只有 support vectors 重要 ───────────────────────────
md(r"""## punchline：整條街只由少數幾個點決定

看那幾個綠圈 —— **只有頂在街緣上的點（support vectors）決定了這條線**。
其他離街很遠的點？把它們移動、甚至刪掉，邊界**完全不動**。

這就是名字的由來：這些點「支撐 (support)」起整個分界，所以叫
**support vector machine**。下面用程式證明給你看：
""")

code(r"""# 找一個「不是 support vector」的點，把它刪掉再訓練一次
clf_full = SVC(kernel="linear", C=1e10).fit(X, y)
sv_idx = set(clf_full.support_)
non_sv = [i for i in range(len(X)) if i not in sv_idx]

drop = non_sv[0]                       # 隨便挑一個非 SV 點
Xd = np.delete(X, drop, axis=0)
yd = np.delete(y, drop)
clf_drop = SVC(kernel="linear", C=1e10).fit(Xd, yd)

print(f"刪掉第 {drop} 號點（非 support vector）後：")
print("  原本   w =", clf_full.coef_[0], " b =", round(clf_full.intercept_[0], 6))
print("  刪點後 w =", clf_drop.coef_[0], " b =", round(clf_drop.intercept_[0], 6))
print("  邊界一樣嗎？", np.allclose(clf_full.coef_, clf_drop.coef_)
      and np.allclose(clf_full.intercept_, clf_drop.intercept_))
""")

# ── 7.5 為什麼街寬 = 2/‖w‖（簡短推導）──────────────────────────
md(r"""## 為什麼街寬剛好是 `2/‖w‖`？

前面那張圖我們是「往兩側加寬到碰最近點」量出街寬。SVM 訓練好後，
街緣其實被釘在兩條平行線上：一邊 `w·x + b = +1`，另一邊 `w·x + b = −1`。
從 `+1` 緣上一點 `x₊` 沿法向量 `w` 方向走垂直距離 `t`，正好踏到 `−1` 緣的點
`x₋ = x₊ − t·(w/‖w‖)`。把兩點各自代入所在緣的式子再相減：

```
w·x₊ + b = +1
w·x₋ + b = −1
─────────────────  相減
w·(x₊ − x₋) = 2
```

而 `x₊ − x₋ = t·(w/‖w‖)`，代進去：`t·(w·w)/‖w‖ = t·‖w‖ = 2`，所以 **`t = 2/‖w‖`**。

`t` 就是兩緣的垂直距離 = **街寬**。因此：

> **街要最寬 ⇔ `‖w‖` 要最小** —— 這就是 SVM 目標寫成 `min ½‖w‖²` 的由來。

> 📖 推導對照 [Python Data Science Handbook — SVM (VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html)
""")

# ── 8. 你的任務（學習模式）─────────────────────────────────────
md(r"""## ✍️ 你的任務：把幾何寫成程式（5 行內）

這是你親手做的部分 —— 也是整課最重要的肌肉記憶。把上面那張圖的三個量公式化：

| 幾何 | 公式 |
|---|---|
| 決策邊界（實線） | `w · x + b = 0` |
| 街的兩個邊緣（虛線） | `w · x + b = ±1` |
| **整條街的寬度** | `2 / ‖w‖` |

**為什麼要你自己寫？** 之後的 soft margin、kernel trick 全部都建在
`f(x) = w·x + b` 這個式子上。親手算一次點積、親手算一次 `2/‖w‖`，
這個地基才真的是你的。

最後再加一題：用你寫的 `decision_value` **自己把 support vectors 抓出來** ——
它們就是頂在街緣、`|f(x)| ≈ 1` 的點。下一格填好**三個**函數，再下一格會**自動對答案**。
""")

if SOLVED:
    decision_body = "    return np.dot(w, x) + b"
    margin_body = "    return 2.0 / np.linalg.norm(w)"
    sv_body = ("    vals = np.array([abs(decision_value(p, w, b)) for p in X])\n"
               "    return np.where(np.abs(vals - 1) < tol)[0]")
else:
    decision_body = "    # TODO: 用 numpy 點積實作 w·x + b（提示：np.dot）\n    raise NotImplementedError(\"換你了！\")"
    margin_body = "    # TODO: 回傳 2 / ‖w‖（提示：np.linalg.norm）\n    raise NotImplementedError(\"換你了！\")"
    sv_body = ("    # TODO: support vector 是「頂在街緣」的點，也就是 |f(x)| ≈ 1\n"
               "    #       用 decision_value 算每個點的 |f|，回傳接近 1 的索引（提示：np.where）\n"
               "    raise NotImplementedError(\"換你了！\")")

code('w = clf.coef_[0]        # 法向量\n'
     'b = clf.intercept_[0]   # 偏置\n\n'
     'def decision_value(x, w, b):\n'
     '    """回傳 f(x) = w·x + b；x 是一個 2D 點，shape (2,)。"""\n'
     + decision_body + '\n\n'
     'def margin_width(w):\n'
     '    """回傳整條街的寬度 = 2 / ‖w‖。"""\n'
     + margin_body + '\n\n'
     'def find_support_vectors(X, w, b, tol=1e-3):\n'
     '    """回傳所有 support vector 的索引（|f(x)| ≈ 1 的點）。"""\n'
     + sv_body + '\n')

# ── 9. 自動對答案 ───────────────────────────────────────────────
md(r"""### 跑這格自動對答案（對照 sklearn）""")

code(r"""test_pt = X[0]
mine = decision_value(test_pt, w, b)
truth = clf.decision_function([test_pt])[0]
ok1 = np.isclose(mine, truth)
print(f"decision_value      : 你的={mine:.4f}  sklearn={truth:.4f}  {'✅' if ok1 else '❌'}")

mine_w = margin_width(w)
ok2 = np.isclose(mine_w, 2 / np.linalg.norm(w))
print(f"margin_width        : 你的={mine_w:.4f}  正解={2/np.linalg.norm(w):.4f}  {'✅' if ok2 else '❌'}")

mine_sv = set(int(i) for i in find_support_vectors(X, w, b))
truth_sv = set(int(i) for i in clf.support_)
ok3 = mine_sv == truth_sv
print(f"find_support_vectors: 你的={sorted(mine_sv)}  sklearn={sorted(truth_sv)}  {'✅' if ok3 else '❌'}")

print("\n整體:", "🎉 通過！你已經把 SVM 的幾何「全部」寫成程式了。" if (ok1 and ok2 and ok3)
      else "再試試 —— 卡住就看下一格提示，或回對話問我。")
""")

# ── 10. 提示 / 解答 ─────────────────────────────────────────────
md(r"""<details>
<summary>👀 卡住了？點開看提示 / 解答</summary>

```python
def decision_value(x, w, b):
    return np.dot(w, x) + b      # 法向量與點的內積，再加偏置

def margin_width(w):
    return 2.0 / np.linalg.norm(w)   # ‖w‖ 越小，街越寬

def find_support_vectors(X, w, b, tol=1e-3):
    vals = np.array([abs(decision_value(p, w, b)) for p in X])
    return np.where(np.abs(vals - 1) < tol)[0]   # 頂在街緣 → |f|≈1
```

**直覺**：街緣是 `w·x+b=±1`，兩條平行線之間的垂直距離正好是 `2/‖w‖`。
所以「最大化間隔」⇔「最小化 `‖w‖`」—— 這就是 SVM 最佳化目標的由來。
</details>
""")

# ── 11. 觀念小測（Feynman 自我檢查）────────────────────────────
md(r"""## 🧠 觀念小測（先想，再點開答案）

1. 把一個**離街很遠**的點往更遠處移動，決策邊界會變嗎？為什麼？
2. SVM 想「最大化間隔」，為什麼數學上寫成「**最小化** `‖w‖`」？
3. 如果兩類資料**根本沒辦法用直線分開**，這一課的 hard-margin 做法會怎樣？

<details>
<summary>點開對答案</summary>

1. **不會**。邊界只由 support vectors（頂在街緣的點）決定；非 SV 點對 `w, b` 沒有貢獻。
2. 街寬 = `2/‖w‖`。要街**寬**，就要 `‖w‖` **小** → 最大化間隔等價於最小化 `‖w‖`（通常寫成 `½‖w‖²`）。
3. 會**找不到解 / 無法滿足「所有點都在街外」的硬限制**。這正是下一課 **soft margin（`C`）** 要解決的問題 —— 允許少數點越界。
</details>
""")

# ── 12. 收尾 + 下一課 ───────────────────────────────────────────
md(r"""## ✅ 這一課你拿到的

- **最大間隔**：在所有能分對的線裡，挑街道最寬的那條 → 泛化最好。
- **support vectors**：只有頂在街緣的少數點決定邊界，其餘點不影響。
- **幾何 ⇄ 程式**：`f(x)=w·x+b`、街寬 `2/‖w‖`、最大化間隔 ⇔ 最小化 `‖w‖`。

## ⏭️ 下一課預告 — Soft Margin 與 `C`
真實資料有雜訊、常常**不可分**。Lesson 2 會放寬「所有點都得在街外」的硬規定，
用一個旋鈕 `C` 來權衡「街要多寬」vs「容許多少越界」（hinge loss 的直覺）。

---

### 📖 延伸閱讀
- [scikit-learn — SVM User Guide](https://scikit-learn.org/stable/modules/svm.html)
- [scikit-learn — SVM Margins Example](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html)
- [Python Data Science Handbook — SVM (VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html)
- [StatQuest — SVM, Clearly Explained!!!](https://statquest.org/support-vector-machines-clearly-explained/)

> 💬 有任何看不懂、想更深、或想我幫你出更難的練習 —— 回對話跟我說。
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

os.makedirs("lessons", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(("[SOLVED] " if SOLVED else "[TODO]   ") + "wrote", OUT, "—", len(cells), "cells")
