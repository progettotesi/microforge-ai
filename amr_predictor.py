import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from data_processor import DataProcessor

class AMRPredictor:
    def __init__(self, model_type='rf', advanced_features=True):
        self.model_type = model_type
        self.advanced_features = advanced_features
        self.processor = DataProcessor()
        
        # Scegli modello di base
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Tipo modello {model_type} non supportato")
        
    def train(self, df):
        """Allena il modello sui dati forniti"""
        # Pre-processing
        if self.advanced_features:
            X, y = self.processor.get_feature_matrix(df)
        else:
            X = df[['mic']]
            y = df['resistant'].astype(int)
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.3, random_state=42
        )
        
        # Training
        self.model.fit(X_train, y_train)
        
        # Valutazione
        train_auc = roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
        
        results = {
            'model': self.model,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'X_test': X_test,
            'y_test': y_test
        }
        
        return results
    
    def predict(self, df):
        """Predizione su nuovi dati"""
        if self.advanced_features:
            X = self.processor.get_feature_matrix(df)
            if isinstance(X, tuple):
                X = X[0]  # Ignora y se restituito
        else:
            X = df[['mic']]
        
        # Probabilità di resistenza
        probs = self.model.predict_proba(X)[:, 1]
        
        # Classe predetta
        preds = self.model.predict(X)
        
        # Risultati con ID
        results = pd.DataFrame({
            'isolate_id': df['isolate_id'] if 'isolate_id' in df.columns else range(len(X)),
            'bacteria': df['bacteria'],
            'antibiotic': df['antibiotic'],
            'mic': df['mic'],
            'resistance_probability': probs,
            'predicted_resistant': preds
        })
        
        return results
    
    def feature_importance(self):
        """Restituisce l'importanza delle features"""
        if not hasattr(self, 'model') or not hasattr(self.processor, 'feature_cols'):
            raise ValueError("Model must be trained first")
        
        importances = pd.DataFrame({
            'feature': self.processor.feature_cols,
            'importance': self.model.feature_importances_
        })
        
        return importances.sort_values('importance', ascending=False)
    
    def save_model(self, filepath):
        """Salva modello su disco"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Carica modello da disco"""
        return joblib.load(filepath)
