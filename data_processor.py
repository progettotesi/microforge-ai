import pandas as pd                                 # Libreria per gestione dati tabellari
import numpy as np                                  # Libreria per operazioni numeriche
from sklearn.preprocessing import StandardScaler    # Per normalizzazione delle feature numeriche
import requests                                     # Per inviare richieste HTTP alle API

# ==========================================
# === CLASSE IMPORT DATI DA API ESTERNE ====
# ==========================================
class DataImporter:
    def __init__(self):
        self.data_sources = {
            'glass': 'https://api.who.int/glass/data',            # WHO GLASS
            'ncbi': 'https://www.ncbi.nlm.nih.gov/pathogens/api', # NCBI Pathogen Detection
            'card': 'https://card.mcmaster.ca/api'                # CARD database
        }

    def import_glass_data(self, pathogen, years=None, countries=None):
        """Importa dati di sorveglianza WHO GLASS tramite API pubblica"""
        params = {"pathogen": pathogen}
        if years:
            params["years"] = ",".join(map(str, years))
        if countries:
            params["countries"] = ",".join(countries)
        url = self.data_sources["glass"]
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            df = pd.json_normalize(data.get("results", data))
            return df
        except Exception as e:
            print(f"Errore importazione WHO GLASS: {e}")
            return pd.DataFrame()

    def import_longitudinal_data(self, source, pathogen, time_period):
        """Importa dati longitudinali da NCBI Pathogen Detection (demo)"""
        if source == "ncbi":
            url = self.data_sources["ncbi"]
            params = {"pathogen": pathogen, "time_period": time_period}
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                df = pd.json_normalize(data.get("results", data))
                return df
            except Exception as e:
                print(f"Errore importazione NCBI: {e}")
                return pd.DataFrame()
        elif source == "card":
            print("L'importazione da CARD richiede autenticazione e parsing custom.")
            return pd.DataFrame()
        else:
            print("Fonte non supportata.")
            return pd.DataFrame()

# ======================================
# === CLASSE PER PROCESSARE I DATI ====
# ======================================
class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()          # Oggetto per normalizzazione standard
        self.feature_cols = []                  # Colonne da usare come input del modello

    def process_data(self, df):
        """
        Processa i dati grezzi e crea features avanzate.
        Integra anche i dati reali dai CSV longitudinali se richiesto.
        """
        processed_df = df.copy()  # Copia per non modificare l'originale

        # Log-transform per valori MIC se presente
        if 'mic' in processed_df.columns:
            processed_df['log_mic'] = np.log1p(processed_df['mic'])

        # One-hot encoding antibiotici
        if 'antibiotic' in processed_df.columns:
            processed_df = pd.concat([
                processed_df, 
                pd.get_dummies(processed_df['antibiotic'], prefix='abx')
            ], axis=1)

        # One-hot encoding batteri
        if 'bacteria' in processed_df.columns:
            processed_df = pd.concat([
                processed_df, 
                pd.get_dummies(processed_df['bacteria'], prefix='bac')
            ], axis=1)

        # Interazioni tra MIC e antibiotici
        if 'antibiotic' in df.columns and 'mic' in df.columns:
            for abx in df['antibiotic'].unique():
                col_name = f'mic_x_{abx}'
                mask = processed_df['antibiotic'] == abx
                processed_df[col_name] = 0
                processed_df.loc[mask, col_name] = processed_df.loc[mask, 'mic']

        # MERGE DATI REALI LONGITUDINALI (tutto il documento)
        try:
            real_paths = [
                "data/DATI REALI/microbiology_cultures_antibiotic_subtype_exposure.csv",
                "data/DATI REALI/microbiology_cultures_cohort.csv",
                "data/DATI REALI/microbiology_cultures_demographics.csv",
                "data/DATI REALI/microbiology_cultures_microbial_resistance.csv",
                "data/DATI REALI/microbiology_cultures_prior_med.csv",
                "data/DATI REALI/microbiology_cultures_vitals.csv"
            ]
            # Carica i CSV e stampa le colonne chiave per debug
            real_dfs = [pd.read_csv(p) for p in real_paths]
            for i, p in enumerate(real_paths):
                print(f"{p}: {real_dfs[i].columns.tolist()}")

            # Merge progressivo solo sulle chiavi comuni effettivamente presenti
            merged = real_dfs[1]
            for rdf in real_dfs[2:]:
                common_keys = [k for k in ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"] if k in merged.columns and k in rdf.columns]
                if not common_keys:
                    print(f"Nessuna chiave comune tra {merged.columns} e {rdf.columns}")
                    continue
                merged = merged.merge(rdf, on=common_keys, how="left")
            # Unisci anche exposure se le chiavi sono presenti
            exposure_keys = [k for k in ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"] if k in merged.columns and k in real_dfs[0].columns]
            if exposure_keys:
                merged = merged.merge(real_dfs[0], on=exposure_keys, how="left")
            else:
                print(f"Nessuna chiave comune tra merged e exposure")
            # Aggiungi colonne reali al processed_df
            for col in merged.columns:
                if col not in processed_df.columns:
                    processed_df[col] = merged[col]
        except Exception as e:
            print(f"Errore merge dati reali: {e}")

        # Salva le feature da usare per il modello (escludendo target e campi categoriali originali)
        self.feature_cols = [c for c in processed_df.columns 
                             if c not in ['resistant', 'bacteria', 'antibiotic']]
        
        return processed_df

    def get_feature_matrix(self, df):
        """
        Estrae la matrice delle feature (e y se presente)
        """
        processed = self.process_data(df)
        X = processed[self.feature_cols]
        if 'resistant' in processed.columns:
            y = processed['resistant'].astype(int)
            return X, y
        else:
            return X

# ====================================================
# === CLASSE ESTESA CON MUTAZIONI E PATHWAY GENOMICI ===
# ====================================================
class EnhancedDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()                           # Eredita tutto da DataProcessor
        self.mutation_db = self._load_mutation_database()  # Carica DB mutazioni
        self.pathway_db = self._load_pathway_database()    # Carica DB pathway

    def _load_mutation_database(self):
        """Carica database di mutazioni note associate a resistenza"""
        return {}

    def _load_pathway_database(self):
        """Carica informazioni sui pathway metabolici rilevanti"""
        return {}

    def process_genomic_data(self, genome_sequence):
        """
        Estrae features genomiche avanzate e annota con CARD/ontologie.
        """
        features = {}
        features['snps'] = self._find_known_snps(genome_sequence)
        features['resistance_genes'] = self._find_resistance_genes(genome_sequence)
        features['pathway_alterations'] = self._analyze_pathways(genome_sequence)
        features['annotations'] = self._annotate_with_card(features['resistance_genes'])
        return features

    def _find_known_snps(self, genome_sequence):
        """Stub per identificazione SNPs noti"""
        # TODO: parsing snps.txt di card-data
        return []

    def _find_resistance_genes(self, genome_sequence):
        """Identifica geni di resistenza noti usando CARD"""
        # Esempio: ricerca di substring dei nomi geni nel file card-data/card.json o .tsv
        card_genes = self._load_card_gene_names()
        found = []
        for gene in card_genes:
            if gene.lower() in genome_sequence.lower():
                found.append(gene)
        return found

    def _analyze_pathways(self, genome_sequence):
        """Stub per analisi pathway metabolici"""
        # TODO: parsing pathway da card-ontology
        return []

    def _load_card_gene_names(self):
        """Carica lista nomi geni di resistenza da CARD (esempio da .tsv)"""
        import os
        card_tsv = os.path.join("data", "DATI REALI", "card-data", "aro_categories.tsv")
        genes = set()
        try:
            import pandas as pd
            df = pd.read_csv(card_tsv, sep="\t")
            if "Name" in df.columns:
                genes.update(df["Name"].dropna().unique())
        except Exception as e:
            print(f"Errore caricamento geni CARD: {e}")
        return genes

    def _annotate_with_card(self, gene_list):
        """Arricchisce i geni trovati con descrizione, meccanismo, antibiotico target, prevalenza"""
        import os
        import pandas as pd
        card_tsv = os.path.join("data", "DATI REALI", "card-data", "aro_categories.tsv")
        prevalence_tsv = os.path.join("data", "DATI REALI", "card-prevalence", "card_prevalence.txt", "card_prevalence.txt")
        annotations = []
        try:
            card_df = pd.read_csv(card_tsv, sep="\t")
        except Exception as e:
            card_df = pd.DataFrame()
        try:
            prev_df = pd.read_csv(prevalence_tsv, sep="\t")
        except Exception as e:
            prev_df = pd.DataFrame()
        for gene in gene_list:
            info = {"gene": gene}
            if not card_df.empty:
                row = card_df[card_df["Name"].str.lower() == gene.lower()]
                if not row.empty:
                    info["description"] = row.iloc[0].get("Description", "")
                    info["mechanism"] = row.iloc[0].get("Resistance Mechanism", "")
                    info["antibiotic"] = row.iloc[0].get("Drug Class", "")
            if not prev_df.empty:
                row = prev_df[prev_df["Gene"].str.lower() == gene.lower()]
                if not row.empty:
                    info["prevalence"] = row.iloc[0].get("Prevalence", "")
            annotations.append(info)
        return annotations

    def create_interaction_features(self, features_dict):
        """Crea features che rappresentano interazioni tra geni"""
        interaction_features = {}
        genes = features_dict.get('resistance_genes', [])
        known_interactions = [('gene1', 'gene2'), ('gene3', 'gene4')]
        for g1, g2 in known_interactions:
            if g1 in genes and g2 in genes:
                interaction_features[f"{g1}_{g2}_interaction"] = 1
            else:
                interaction_features[f"{g1}_{g2}_interaction"] = 0
        return interaction_features

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
