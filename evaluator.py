from model import train_model
from sklearn.metrics import roc_auc_score
import pandas as pd

def evaluate(_):
    df = pd.read_csv("data/amr_dataset.csv")
    model, X_test, y_test = train_model(df)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return {"auc": auc}
