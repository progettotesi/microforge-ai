import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = []
    
    def process_data(self, df):
        """
        Processa i dati grezzi e crea features avanzate
        """
        # Copia per non modificare l'originale
        processed_df = df.copy()
        
        # Features avanzate
        # 1. Log-transform delle MIC (importante per valori distribuiti esponenzialmente)
        processed_df['log_mic'] = np.log1p(processed_df['mic'])
        
        # 2. Features specifiche degli antibiotici
        processed_df = pd.concat([
            processed_df, 
            pd.get_dummies(processed_df['antibiotic'], prefix='abx')
        ], axis=1)
        
        # 3. Features specifiche dei batteri
        processed_df = pd.concat([
            processed_df, 
            pd.get_dummies(processed_df['bacteria'], prefix='bac')
        ], axis=1)
        
        # 4. Interazione MIC - antibiotico
        for abx in df['antibiotic'].unique():
            col_name = f'mic_x_{abx}'
            mask = processed_df['antibiotic'] == abx
            processed_df[col_name] = 0
            processed_df.loc[mask, col_name] = processed_df.loc[mask, 'mic']
        
        # Rimuovi colonne non necessarie per il modello SOLO SE non serve per evoluzione
        # (lascia 'isolate_id' se presente, sarà usata da ResistanceEvolutionModel)
        # drop_cols = ['isolate_id']
        # processed_df = processed_df.drop(drop_cols, axis=1, errors='ignore')
        
        # Salva colonne features per futuri riferimenti
        self.feature_cols = [c for c in processed_df.columns 
                           if c not in ['resistant', 'bacteria', 'antibiotic']]
        
        return processed_df
    
    def scale_features(self, X):
        """
        Normalizza le features numeriche
        """
        numeric_cols = X.select_dtypes(include=['float', 'int']).columns
        X_scaled = X.copy()
        X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        return X_scaled
    
    def get_feature_matrix(self, df):
        """
        Estrae la matrice di features per il modello
        """
        processed = self.process_data(df)
        X = processed[self.feature_cols]
        if 'resistant' in processed.columns:
            y = processed['resistant'].astype(int)
            return X, y
        else:
            return X
