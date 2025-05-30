import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="MICROFORGE AI", layout="centered")
st.title("🧬 MICROFORGE AI")
st.subheader("Previsione della resistenza antibiotica con AI evolutiva")

uploaded_file = st.file_uploader("📁 Carica un file CSV con i dati AMR", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File caricato correttamente")
    st.write("📊 Anteprima dati:", df.head())

    if st.button("▶️ Avvia valutazione modello"):
        with st.spinner("⏳ Elaborazione in corso..."):
            df["resistant"] = df["resistant"].astype(int)
            X = df[["mic"]]
            y = df["resistant"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            st.success("✅ Modello valutato correttamente")
            st.metric("AUC", round(auc, 4))
