import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def evaluate():
    df = pd.read_csv("data/amr_dataset.csv")  # oppure usa `uploaded_file` se vuoi passarlo
    df["resistant"] = df["resistant"].astype(int)

    X = df[["mic"]]
    y = df["resistant"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    return {"auc": auc}
