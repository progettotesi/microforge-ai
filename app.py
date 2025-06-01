# IMPORTAZIONE LIBRERIE E MODULI NECESSARI


import streamlit as st                # Libreria principale per creare l'interfaccia web
import pandas as pd                  # Per gestire i dataset (CSV)
import numpy as np                   # Per operazioni numeriche
import os                            # Per gestire file di sistema
import time                          # Per ritardi e temporizzazioni
import asyncio                       # Per gestire chiamate asincrone (es. OpenEvolve)
import plotly.express as px          # Grafici interattivi (es. mappe, linee, barre)
import plotly.graph_objects as go    # Grafici avanzati (più personalizzazione)
from amr_predictor import AMRPredictor                      # Classe per la predizione AMR
from resistance_evolution import ResistanceEvolutionModel   # Classe per l’evoluzione AMR
from visualization import create_resistance_dashboard       # Funzione per generare dashboard finale
from evolve_model import run_open_evolve                    # Funzione per attivare AI evolutiva OpenEvolve


# CONFIGURAZIONE INZIALE DELL'INTERFACCIA

# Configurazione pagina
st.set_page_config(page_title="MICROFORGE AI", layout="wide")  # Titolo e layout largo
st.title("🧬 MICROFORGE AI™")                                    # Titolo principale
st.subheader("Predictive Evolution & Resistance Engine for Pathogens")  # Sottotitolo

# SIDEBAR: Opzioni e Caricamento (a sinistra)
with st.sidebar:
    st.image("logo/logosolom.png", width=300)  # Logo a sinistra
    st.header("🧠 MICROFORGE AI™")
    
    # Opzioni modello
    st.subheader("Opzioni Modello")       # Sezione modello predittivo
    model_type = st.selectbox(
        "Algoritmo di base",
        ["RandomForest", "GradientBoosting"]
    )

    use_advanced = st.checkbox("Usa Features Avanzate", value=True)   # check opzioni avanzate

    # Opzioni evoluzione
    st.subheader("Opzioni Evoluzione")
    months_ahead = st.slider("Mesi di Previsione", 1, 12, 6)     # Quanto in là prevedere l'evoluzione
    evolve_engine = st.selectbox("Motore di Evoluzione", ["OpenEvolve", "MiniAlphaEvolve"])



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
if df is not None:                                      # Se l'utente ha caricato un dataset valido
    st.markdown("---")
    st.header("🔬 Analisi Predittiva")

# Crea 3 tab:Resistenza attuale,Evoluzione della resistenza,Analisi avanzata con OpenEvolve AI

    analysis_tab, evolution_tab, advanced_tab = st.tabs([                     
        "🔬 Resistenza Attuale", 
        "🧬 Evoluzione Resistenza", 
        "🧠 Analisi Avanzata"
    ])
    
    with analysis_tab:
        if st.button("▶️ Avvia Analisi Resistenza"):
            with st.spinner("⏳ Elaborazione predizione resistenza..."):

# Crea e allena modello , Crea un oggetto AMRPredictor con: # Algoritmo scelto:
#  RandomForest (rf) o GradientBoosting (gb) e Uso di feature avanzate opzionali

                predictor = AMRPredictor(
                    model_type='rf' if model_type == "RandomForest" else 'gb',
                    advanced_features=use_advanced
                )
                
                # Addestra il modello sui dati caricati (df).

                model_results = predictor.train(df)
                
                # Statistiche modello
                st.subheader("📈 Performance Modello")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AUC Training", f"{model_results['train_auc']:.4f}")
                with col2:
                    st.metric("AUC Testing", f"{model_results['test_auc']:.4f}")
                
                # Il modello produce predizioni e le mostra in tabella. 
                predictions = predictor.predict(df)
                st.subheader("🎯 Predizioni Resistenza")
                st.dataframe(predictions)
                
                # Mostra un grafico a barre con le feature più rilevanti.
                st.subheader("🧬 Features Importanti")
                importance = predictor.feature_importance()
                st.bar_chart(importance.set_index('feature')['importance'])
                
                # Salva risultati in sessione
                st.session_state.predictions = predictions
                st.session_state.importance = importance
                st.session_state.predictor = predictor
                
                st.success("✅ Analisi completata con successo")
    


# TAB PER L'EVOLUZIONE DELLA RESISTENZA

# Quando l’utente preme il bottone, si simula l’evoluzione futura della resistenza.    
    with evolution_tab:
        if st.button("🔮 Avvia Predizione Evoluzione"):
            with st.spinner("⏳ Elaborazione predizione evoluzione..."):
                # Riutilizza il modello già addestrato, altrimenti lo ricrea e salva in sessione.
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
                
                # Utilizza solo dati reali caricati dall'utente, senza simulazione di dati longitudinali
                current_data = df.copy()
                # Assicura che le colonne chiave siano presenti ma non crea dati fittizi
                for col in ['isolate_id', 'antibiotic']:
                    if col not in current_data.columns:
                        st.error(f"Colonna mancante: {col}. Il file deve contenere tutte le colonne richieste.")
                        st.stop()
                
                # Modello evoluzione
                evolution_model = ResistanceEvolutionModel(
                    time_horizon=months_ahead,
                    model_type='rf' if model_type == "RandomForest" else 'gb'
                )
                
                # Allena il modello di evoluzione sui dati reali caricati dall'utente
                evolution_model.train(current_data, real_training=True)
                
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
        if st.button("🔬 Avvia Evoluzione AI"):
            with st.spinner("🧠 Evoluzione del modello in corso..."):
                try:
                    if evolve_engine == "OpenEvolve":
                        metrics = asyncio.run(run_open_evolve())
                        st.success("✅ Evoluzione AI completata con OpenEvolve")
                    else:
                        from mini_alphaevolve_runner import run_mini_alpha_evolve  # vedremo sotto come crearlo
                        metrics = asyncio.run(run_mini_alpha_evolve())
                        st.success("✅ Evoluzione AI completata con MiniAlphaEvolve")

                    st.subheader("📊 Risultati Evolutivi")
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

class EnhancedDashboard:
    def __init__(self):
        self.tabs = ["Overview", "Predizione", "Evoluzione", "Genomica", "Epidemiologia"]
    
    def render_epidemiology_tab(self, data):
        """Crea visualizzazioni epidemiologiche interattive e confronto con dati European"""
        st.header("🔬 Analisi Epidemiologica")

        import os
        import pandas as pd

        # Carica tutti i file .xlsx dalla cartella European ricorsivamente
        european_dir = os.path.join("data", "DATI REALI", "European")
        euro_files = []
        for root, dirs, files in os.walk(european_dir):
            for file in files:
                if file.lower().endswith(".xlsx"):
                    euro_files.append(os.path.join(root, file))

        # Permetti selezione file epidemiologico
        if euro_files:
            st.subheader("📊 Seleziona dataset epidemiologico europeo")
            selected_file = st.selectbox("Scegli file", euro_files)
            try:
                df_euro = pd.read_excel(selected_file)
                st.write("Anteprima dati European:", df_euro.head())
                # Selezione colonne chiave per confronto
                columns = df_euro.columns.tolist()
                if len(columns) >= 3:
                    x_col = st.selectbox("Colonna X (es. anno, paese)", columns, index=0)
                    y_col = st.selectbox("Colonna Y (es. % resistenza)", columns, index=1)
                    group_col = st.selectbox("Colonna Gruppo (es. batterio, antibiotico)", columns, index=2)
                    # Grafico interattivo
                    fig = px.line(
                        df_euro,
                        x=x_col,
                        y=y_col,
                        color=group_col,
                        title=f"Andamento {y_col} per {group_col}"
                    )
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Errore caricamento dati European: {e}")

        # Se disponibili dati locali, confronto con European
        if 'time_series' in data:
            st.subheader("📈 Confronto temporale locale vs European")
            fig = px.line(
                data['time_series'],
                x='date',
                y='resistance_rate',
                color='antibiotic',
                line_group='bacteria',
                hover_name='bacteria',
                title='Evoluzione Temporale della Resistenza (locale)'
            )
            st.plotly_chart(fig)
    
    def render_genomic_analyzer(self, model):
        """Interfaccia per analisi genomica"""
        st.header("🧬 Analizzatore Genomico")
        
        # Upload file FASTA
        uploaded_file = st.file_uploader(
            "Carica sequenza genomica in formato FASTA", 
            type=['fasta', 'fa', 'fna']
        )
        
        if uploaded_file is not None:
            # Processo di analisi
            with st.spinner("Analisi genomica in corso..."):
                # Leggi la sequenza dal file FASTA
                genome_sequence = ""
                for line in uploaded_file:
                    line = line.decode("utf-8")
                    if not line.startswith(">"):
                        genome_sequence += line.strip()
                # Analisi con EnhancedDataProcessor
                from data_processor import EnhancedDataProcessor
                processor = EnhancedDataProcessor()
                features = processor.process_genomic_data(genome_sequence)
                # Mostra risultati
                st.subheader("🧬 Geni di resistenza trovati")
                if features["annotations"]:
                    st.dataframe(features["annotations"])
                else:
                    st.info("Nessun gene di resistenza noto trovato nella sequenza.")
                # Mostra anche SNPs e pathway se disponibili
                if features["snps"]:
                    st.subheader("🔬 SNPs noti trovati")
                    st.write(features["snps"])
                if features["pathway_alterations"]:
                    st.subheader("🧬 Alterazioni pathway metabolici")
                    st.write(features["pathway_alterations"])
