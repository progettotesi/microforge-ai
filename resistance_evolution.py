import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt
from data_processor import DataProcessor

class ResistanceEvolutionModel:
    def __init__(self, time_horizon=6, model_type='rf'):
        """
        Modello per prevedere l'evoluzione della resistenza nel tempo
        
        Args:
            time_horizon: Orizzonte di predizione in mesi
            model_type: 'rf' (RandomForest) o 'gb' (GradientBoosting)
        """
        self.time_horizon = time_horizon
        self.processor = DataProcessor()
        
        # Modello per predirre l'incremento MIC nel tempo
        if model_type == 'rf':
            self.evolution_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'gb':
            self.evolution_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Tipo modello {model_type} non supportato per evoluzione")
        
        # Soglie MIC per antibiotici comuni
        self.resistance_breakpoints = {
            'Cefepime': 8,      # mcg/mL
            'Ciprofloxacin': 1,  # mcg/mL
            'Meropenem': 4      # mcg/mL
        }
    
    def train(self, longitudinal_df):
        """
        Allena il modello su dati longitudinali
        
        Args:
            longitudinal_df: DataFrame con colonne isolate_id, bacteria, antibiotic, 
                            mic, time_point (mesi)
        """
        # Preprocessing
        X = self.processor.process_data(longitudinal_df)
        
        # Preparazione target: Incremento MIC al tempo t+1
        X['next_mic'] = X.groupby(['isolate_id', 'antibiotic'])['mic'].shift(-1)
        X['mic_change'] = X['next_mic'] - X['mic']
        
        # Rimuovi righe senza punto successivo
        X = X.dropna(subset=['next_mic', 'mic_change'])
        
        # Prepara features e target
        features = [c for c in X.columns if c not in 
                   ['resistant', 'isolate_id', 'mic_change', 'next_mic']]
        
        X_train = X[features]
        y_train = X['mic_change']

        # Usa solo colonne numeriche per il modello
        X_train = X_train.select_dtypes(include=[np.number])
        # Salva le colonne numeriche usate per il fit
        self.feature_names_ = X_train.columns.tolist()
        
        # Controllo dati sufficienti
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Dati insufficienti per l'addestramento del modello di evoluzione. Servono dati longitudinali con almeno due time point per ogni isolate_id e antibiotic.")
        
        # Allena modello
        self.evolution_model.fit(X_train, y_train)
        
        # Valutazione
        pred_changes = self.evolution_model.predict(X_train)
        mse = np.mean((y_train - pred_changes)**2)
        
        return {'mse': mse, 'model': self.evolution_model}
    
    def predict_evolution(self, df, months_ahead=6):
        """
        Predice l'evoluzione della MIC nei mesi successivi
        
        Args:
            df: DataFrame con dati attuali
            months_ahead: Numero mesi da predire
        
        Returns:
            DataFrame con predizioni MIC e resistenza
        """
        results = []
        
        # Per ogni isolato
        for _, row in df.iterrows():
            isolate_id = row['isolate_id'] if 'isolate_id' in df.columns else 'unknown'
            bac = row['bacteria']
            abx = row['antibiotic']
            current_mic = row['mic']
            
            # Simulazione evoluzione
            mic_values = [current_mic]
            current_data = pd.DataFrame({
                'isolate_id': [isolate_id],
                'bacteria': [bac],
                'antibiotic': [abx],
                'mic': [current_mic]
            })
            
            # Progressivamente predici cambiamenti MIC
            for month in range(1, months_ahead + 1):
                # Processo dato corrente
                X = self.processor.process_data(current_data)
                feature_cols = [c for c in X.columns if c not in ['resistant', 'isolate_id']]
                # Usa solo le colonne numeriche viste nel fit
                # Usa solo le colonne numeriche viste nel fit, aggiungendo colonne mancanti con 0
                if hasattr(self, "feature_names_"):
                    for col in self.feature_names_:
                        if col not in X.columns:
                            X[col] = 0
                    X_pred = X[self.feature_names_]
                else:
                    X_pred = X.select_dtypes(include=[np.number])
                # Predici cambiamento MIC
                mic_change = self.evolution_model.predict(X_pred)[0]
                new_mic = max(0.001, current_mic + mic_change)  # Minimo MIC
                
                # Ottieni breakpoint per antibiotico o default
                if abx in self.resistance_breakpoints:
                    breakpoint = self.resistance_breakpoints[abx]
                else:
                    breakpoint = 4  # Valore di default
                
                # Determina se è resistente
                resistant = 1 if new_mic >= breakpoint else 0
                
                # Salva risultato
                results.append({
                    'isolate_id': isolate_id,
                    'bacteria': bac,
                    'antibiotic': abx,
                    'month': month,
                    'predicted_mic': new_mic,
                    'predicted_resistant': resistant,
                    'resistance_probability': 1 if new_mic >= breakpoint * 1.5 else 
                                             (0 if new_mic < breakpoint * 0.5 else 
                                              (new_mic - breakpoint * 0.5) / breakpoint)
                })
                
                # Aggiorna MIC corrente per prossima iterazione
                current_mic = new_mic
                current_data = pd.DataFrame({
                    'isolate_id': [isolate_id],
                    'bacteria': [bac],
                    'antibiotic': [abx],
                    'mic': [current_mic]
                })
        
        return pd.DataFrame(results)
    
    def plot_resistance_evolution(self, evolution_df):
        """
        Crea grafici per l'evoluzione della resistenza
        
        Args:
            evolution_df: DataFrame risultante da predict_evolution
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Per ogni combinazione batterio-antibiotico
        for (bac, abx), group in evolution_df.groupby(['bacteria', 'antibiotic']):
            group = group.sort_values('month')
            ax.plot(group['month'], group['predicted_mic'], 
                    marker='o', label=f"{bac} - {abx}")
            
            # Aggiungi linea soglia
            if abx in self.resistance_breakpoints:
                breakpoint = self.resistance_breakpoints[abx]
                ax.axhline(y=breakpoint, color='r', linestyle='--', alpha=0.5)
            
        ax.set_title("Evoluzione della Resistenza Antimicrobica")
        ax.set_xlabel("Mesi")
        ax.set_ylabel("MIC predetta")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scala logaritmica per MIC
        ax.set_yscale('log')
        
        return fig
