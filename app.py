import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import asyncio
from amr_predictor import AMRPredictor
from resistance_evolution import ResistanceEvolutionModel
from visualization import create_resistance_dashboard
from evolve_model import run_open_evolve

# Configurazione pagina
st.set_page_config(page_title="MICROFORGE AI", layout="wide")

# Header app
st.title("🧬 MICROFORGE AI™")
st.subheader("Predictive Evolution & Resistance Engine for Pathogens")

# Sidebar
with st.sidebar:
    st.image("logo/logo.png", width=100)
    st.header("🧠 MICROFORGE AI™")
    
    # Opzioni modello
    st.subheader("Opzioni Modello")
    model_type = st.selectbox(
        "Algoritmo di base",
        ["RandomForest", "GradientBoosting"]
    )
    
    use_advanced = st.checkbox("Usa Features Avanzate", value=True)
    
    # Opzioni evoluzione
    st.subheader("Opzioni Evoluzione")
    months_ahead = st.slider("Mesi di Previsione", 1, 12, 6)
    
    # About info
    st.markdown("---")
    st.caption("© 2025 MICROFORGE AI™")
    st.caption("Predictive Evolution & Resistance Engine")

# Divisione pagina principale
upload_col, info_col = st.columns([2,1])

# Colonna upload file
with upload_col:
    st.header("📁 Carica Dati")
    uploaded_file = st.file_uploader("📊 Carica un file CSV con i dati AMR", type="csv")
    
    # Sample dataset
    if not uploaded_file and st.button("🔬 Usa Dataset di Esempio"):
        df = pd.read_csv("data/amr_dataset.csv")
        st.success("✅ Dataset di esempio caricato")
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.to_csv("data/uploaded.csv", index=False)  # Salva il CSV per l'uso interno
        st.success("✅ File caricato correttamente")
    else:
        df = None
    
    # Preview dati
    if df is not None:
        st.write("📊 Anteprima dati:", df.head())
        
        # Statistiche base
        st.subheader("📈 Statistiche Dataset")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🦠 Batteri Unici", len(df['bacteria'].unique()))
        
        with col2:
            st.metric("💊 Antibiotici Unici", len(df['antibiotic'].unique()))
        
        with col3:
            st.metric("📝 Isolati Totali", df['isolate_id'].nunique() if 'isolate_id' in df.columns else len(df))

# Colonna info
with info_col:
    st.header("ℹ️ MICROFORGE AI")
    st.write("""
    **MICROFORGE AI™** utilizza algoritmi di intelligenza artificiale evoluzionaria per:
    
    1. **Predirre resistenza antimicrobica** attuale
    2. **Prevedere evoluzione** futura dei patogeni
    3. **Identificare rischi** emergenti
    
    Carica il tuo dataset e scopri l'evoluzione della resistenza!
    """)
    
    st.info("📋 **Formato richiesto CSV:**  \nIl file deve contenere: isolate_id, bacteria, antibiotic, mic, resistant")

# Area analisi (se dati disponibili)
if df is not None:
    st.markdown("---")
    st.header("🔬 Analisi Predittiva")
    
    analysis_tab, evolution_tab, advanced_tab = st.tabs([
        "🔬 Resistenza Attuale", 
        "🧬 Evoluzione Resistenza", 
        "🧠 Analisi Avanzata"
    ])
    
    with analysis_tab:
        if st.button("▶️ Avvia Analisi Resistenza"):
            with st.spinner("⏳ Elaborazione predizione resistenza..."):
                # Crea e allena modello
                predictor = AMRPredictor(
                    model_type='rf' if model_type == "RandomForest" else 'gb',
                    advanced_features=use_advanced
                )
                
                # Allena modello
                model_results = predictor.train(df)
                
                # Statistiche modello
                st.subheader("📈 Performance Modello")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AUC Training", f"{model_results['train_auc']:.4f}")
                with col2:
                    st.metric("AUC Testing", f"{model_results['test_auc']:.4f}")
                
                # Predizioni
                predictions = predictor.predict(df)
                st.subheader("🎯 Predizioni Resistenza")
                st.dataframe(predictions)
                
                # Feature importance
                st.subheader("🧬 Features Importanti")
                importance = predictor.feature_importance()
                st.bar_chart(importance.set_index('feature')['importance'])
                
                # Salva risultati in sessione
                st.session_state.predictions = predictions
                st.session_state.importance = importance
                st.session_state.predictor = predictor
                
                st.success("✅ Analisi completata con successo")
    
    with evolution_tab:
        if st.button("🔮 Avvia Predizione Evoluzione"):
            with st.spinner("⏳ Elaborazione predizione evoluzione..."):
                # Crea modello evoluzione
                if not 'predictor' in st.session_state:
                    # Crea e allena modello se non esistente
                    predictor = AMRPredictor(
                        model_type='rf' if model_type == "RandomForest" else 'gb',
                        advanced_features=use_advanced
                    )
                    model_results = predictor.train(df)
                    predictions = predictor.predict(df)
                    st.session_state.predictions = predictions
                    st.session_state.predictor = predictor
                
                # Simula dati longitudinali (in un caso reale avresti dati storici)
                # Questo è solo per simulazione, idealmente avresti dati reali
                current_data = df.copy()
                # Assicura che le colonne chiave siano presenti
                for col in ['isolate_id', 'antibiotic']:
                    if col not in current_data.columns:
                        current_data[col] = [f"sample_{i}" for i in range(len(current_data))] if col == 'isolate_id' else 'unknown'
                
                # Modello evoluzione
                evolution_model = ResistanceEvolutionModel(
                    time_horizon=months_ahead,
                    model_type='rf' if model_type == "RandomForest" else 'gb'
                )
                
                # Allena il modello di evoluzione (training fittizio sui dati attuali)
                evolution_model.train(current_data)
                
                # Genera simulazione evoluzione
                evolution_results = evolution_model.predict_evolution(current_data, months_ahead=months_ahead)
                
                # Visualizzazioni
                st.subheader("📈 Previsione Evoluzione Resistenza")
                fig = evolution_model.plot_resistance_evolution(evolution_results)
                st.pyplot(fig)
                
                # Tabella evoluzione
                st.subheader("📊 Dettaglio Evoluzione")
                st.dataframe(evolution_results)
                
                # Alert critici
                critical = evolution_results[
                    (evolution_results['predicted_resistant'] == 1) & 
                    (evolution_results['month'] <= 3)
                ]
                
                if not critical.empty:
                    st.warning("⚠️ **ALERT: Evoluzione Critica Rilevata**")
                    for _, row in critical.iterrows():
                        st.error(f"⚠️ {row['bacteria']} diventerà resistente a {row['antibiotic']} entro {row['month']} mesi!")
                
                # Salva risultati in sessione
                st.session_state.evolution = evolution_results
    
    with advanced_tab:
        if st.button("🔬 Avvia OpenEvolve AI"):
            with st.spinner("🧠 Evoluzione del modello con AI..."):
                try:
                    # Esegui OpenEvolve
                    metrics = asyncio.run(run_open_evolve())
                    st.success("✅ Evoluzione AI completata")
                    
                    # Mostra metriche ottenute
                    st.subheader("📊 Risultati OpenEvolve AI")
                    for name, value in metrics.items():
                        st.metric(f"🔬 {name}", round(value, 4))
                    
                except Exception as e:
                    st.error(f"Errore durante l'evoluzione: {e}")
                    
        # Dashboard completa (se disponibili tutti i dati)
        if ('predictions' in st.session_state and 
            'evolution' in st.session_state and 
            'importance' in st.session_state):
            
            st.subheader("📊 Dashboard Completa")
            if st.button("📊 Genera Dashboard"):
                create_resistance_dashboard(
                    st.session_state.predictions,
                    st.session_state.evolution,
                    st.session_state.importance
                )
