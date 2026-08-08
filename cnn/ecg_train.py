"""ECG (MIT-BIH) 5-class classification -> Kaggle submission.

Data: cnn/ecg_data/kaggle/{train,test}.csv from the course Google Drive.
train: 187 signal cols (0..186) + label col (187). test: 187 signal cols only.
Submission: Id (000-indexed, 3-digit), Category (0~4). Baseline to beat: 0.86027.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

HERE = Path(__file__).parent
DATA = HERE / "ecg_data" / "kaggle"
OUT = HERE / "AIMD_BA28-11_HW.csv"

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

X = train.iloc[:, :187].values          # signal columns 0..186
y = train.iloc[:, 187].values.astype(int)
Xtest = test.iloc[:, :187].values       # test has no label column
assert Xtest.shape[1] == 187, Xtest.shape

clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)
cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print(f"5-fold CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}  (baseline 0.86027)")

clf.fit(X, y)
pred = clf.predict(Xtest).astype(int)

sub = pd.DataFrame({
    "Id": [f"{i:03d}" for i in range(len(pred))],
    "Category": pred,
})
sub.to_csv(OUT, index=False)
print(f"wrote {OUT}  shape={sub.shape}")
print("pred distribution:", dict(zip(*np.unique(pred, return_counts=True))))
print(sub.head().to_string(index=False))
