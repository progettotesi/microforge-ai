import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plot_resistance_heatmap(df):
    """
    Crea heatmap delle probabilità di resistenza
    """
    # Pivot della tabella
    pivot = df.pivot_table(
        index='bacteria', 
        columns='antibiotic', 
        values='resistance_probability',
        aggfunc='mean'
    )
    
    # Plot con Plotly
    fig = px.imshow(
        pivot, 
        text_auto='.2f',
        color_continuous_scale='RdYlGn_r',
        labels=dict(x="Antibiotico", y="Batterio", color="Prob. Resistenza"),
        title="Probabilità di Resistenza per Batterio-Antibiotico"
    )
    
    fig.update_layout(width=700, height=500)
    return fig

def plot_evolution_forecast(evolution_df, antibiotic=None):
    """
    Crea grafico interattivo dell'evoluzione resistenza
    """
    if antibiotic:
        filtered_df = evolution_df[evolution_df['antibiotic'] == antibiotic]
    else:
        filtered_df = evolution_df
    
    fig = px.line(
        filtered_df, 
        x='month', 
        y='predicted_mic', 
        color='bacteria',
        line_dash='antibiotic',
        markers=True,
        hover_data=['predicted_resistant', 'resistance_probability'],
        labels={'predicted_mic': 'MIC Prevista', 'month': 'Mesi'},
        title=f"Previsione Evoluzione Resistenza {'per ' + antibiotic if antibiotic else ''}"
    )
    
    # Aggiungi linee soglia
    breakpoints = {
        'Cefepime': 8,
        'Ciprofloxacin': 1,
        'Meropenem': 4
    }
    
    for abx, bp in breakpoints.items():
        if antibiotic and antibiotic != abx:
            continue
            
        fig.add_shape(
            type="line",
            x0=0,
            y0=bp,
            x1=filtered_df['month'].max(),
            y1=bp,
            line=dict(color="red", width=1, dash="dash"),
            name=f"Breakpoint {abx}"
        )
    
    # Scala logaritmica per MIC
    fig.update_layout(
        yaxis_type="log",
        width=800,
        height=500,
        xaxis_title="Mesi",
        yaxis_title="MIC (log scale)",
        legend_title="Batterio - Antibiotico"
    )
    
    return fig

def plot_feature_importance(importance_df, top_n=15):
    """
    Crea grafico delle feature importance
    """
    # Prendi top N features
    top_features = importance_df.head(top_n)
    
    fig = px.bar(
        top_features, 
        x='importance', 
        y='feature',
        orientation='h',
        title=f"Top {top_n} Features Importanti",
        labels={'importance': 'Importanza', 'feature': 'Feature'}
    )
    
    fig.update_layout(width=700, height=500)
    return fig

def create_resistance_dashboard(current_df, evolution_df, importance_df):
    """
    Crea dashboard completa per resistenza
    """
    st.header("📊 Dashboard Previsionale Resistenza")
    
    # Tab per diverse visualizzazioni
    tab1, tab2, tab3 = st.tabs(["📈 Stato Attuale", "🔮 Evoluzione", "🧬 Feature Importanti"])
    
    with tab1:
        st.subheader("Stato Attuale Resistenza")
        heatmap = plot_resistance_heatmap(current_df)
        st.plotly_chart(heatmap)
        
        # Tabella dettagliata
        st.dataframe(current_df)
    
    with tab2:
        st.subheader("Previsione Evoluzione Resistenza")
        
        # Filtro per antibiotico
        antibiotics = sorted(evolution_df['antibiotic'].unique())
        selected_abx = st.selectbox("Seleziona Antibiotico", ['Tutti'] + list(antibiotics))
        
        # Plot evoluzione
        evo_plot = plot_evolution_forecast(
            evolution_df, 
            antibiotic=selected_abx if selected_abx != 'Tutti' else None
        )
        st.plotly_chart(evo_plot)
        
        # Tabella dettagliata filtrata
        if selected_abx != 'Tutti':
            filtered_evo = evolution_df[evolution_df['antibiotic'] == selected_abx]
        else:
            filtered_evo = evolution_df
        
        st.dataframe(filtered_evo)
        
        # Alert per previsioni critiche
        critical = evolution_df[
            (evolution_df['predicted_resistant'] == 1) & 
            (evolution_df['month'] <= 3)
        ]
        
        if not critical.empty:
            st.warning("⚠️ **ALERT: Evoluzione Critica Rilevata**")
            for _, row in critical.iterrows():
                st.error(f"⚠️ {row['bacteria']} diventerà resistente a {row['antibiotic']} entro {row['month']} mesi!")
    
    with tab3:
        st.subheader("Feature Importanti per la Previsione")
        importance_plot = plot_feature_importance(importance_df)
        st.plotly_chart(importance_plot)
        
        # Insights sulle features
        st.info("ℹ️ La MIC attuale è il fattore più predittivo per l'evoluzione della resistenza")
        
        # Tabella completa
        st.dataframe(importance_df)
