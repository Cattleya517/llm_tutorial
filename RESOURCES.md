# Support Vector Machines (SVM) Resources

## Knowledge

- [scikit-learn — 1.4. Support Vector Machines (User Guide)](https://scikit-learn.org/stable/modules/svm.html)
  官方文件。最大間隔 = 最小化 ‖w‖²、soft margin 的 `C`、各 kernel 的 API。Use for: 實作參數對照、`kernel`/`C`/`gamma` 的權威定義。
- [scikit-learn — SVM Margins Example](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html)
  官方範例：畫出決策邊界、margin 邊界與 support vectors。Use for: 視覺化範本。
- [Python Data Science Handbook — In-Depth: Support Vector Machines (VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html)
  最佳的「最寬街道」直覺講解 + 可跑程式。Use for: maximum margin 直覺、support vector 只看邊界點的示範。**本課 2D 範例即取自此。**
- [StatQuest — Support Vector Machines, Clearly Explained!!! (Josh Starmer)](https://statquest.org/support-vector-machines-clearly-explained/)
  三部曲影片：maximal margin → soft margin → kernel trick。Use for: 純直覺、無痛入門的影片補充。
- [KDnuggets — A Gentle Introduction to Support Vector Machines](https://www.kdnuggets.com/2023/07/gentle-introduction-support-vector-machines.html)
  maximum margin classifier 概念的簡潔導讀。Use for: 名詞快速釐清。

## Wisdom (Communities)

- [Cross Validated — `svm` 標籤 (stats.stackexchange.com)](https://stats.stackexchange.com/questions/tagged/svm) / [r/MachineLearning](https://reddit.com/r/MachineLearning)
  Use for: 觀念釐清、為什麼某個 kernel/`C` 在你的資料上表現如此。
  （使用者尚未表達是否想加入社群 —— 之後再確認偏好。）

## Gaps
- 尚缺一個「直覺優先」但有完整 dual / KKT 推導的**中文**資源；若之後想深入數學，需再找（Stanford CS229 講義為英文候選）。
