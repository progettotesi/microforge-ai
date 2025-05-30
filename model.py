import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# EVOLVE-BLOCK-START
def train_model(df):
    df["resistant"] = df["resistant"].astype(int)
    X = df[["mic"]]
    y = df["resistant"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test
# EVOLVE-BLOCK-END
