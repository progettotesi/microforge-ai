import pandas as pd
import numpy as np
import random

def generate_amr_dataset(num_isolates=100, save_path="data/amr_dataset_extended.csv"):
    """Genera dataset AMR più completo e realistico"""
    
    # Definizioni base
    bacteria = ["P.aeruginosa", "E.coli", "K.pneumoniae", "S.aureus", "A.baumannii"]
    
    antibiotics = {
        "P.aeruginosa": ["Cefepime", "Ciprofloxacin", "Meropenem", "Piperacillin-tazobactam", "Colistin"],
        "E.coli": ["Ampicillin", "Ciprofloxacin", "Ceftriaxone", "Meropenem", "Gentamicin"],
        "K.pneumoniae": ["Ceftriaxone", "Ciprofloxacin", "Meropenem", "Amikacin", "Tigecycline"],
        "S.aureus": ["Oxacillin", "Vancomycin", "Clindamycin", "Daptomycin", "Linezolid"],
        "A.baumannii": ["Colistin", "Cefepime", "Meropenem", "Tigecycline", "Gentamicin"]
    }
    
    data = []
    for i in range(num_isolates):
        bac = random.choice(bacteria)
        abx = random.choice(antibiotics[bac])
        isolate_id = f"{i+1:03d}"
        mic = np.round(np.random.lognormal(mean=1.5, sigma=1.0), 2)
        resistant = int(mic > 4)
        data.append([isolate_id, bac, abx, mic, resistant])
    
    df = pd.DataFrame(data, columns=["isolate_id", "bacteria", "antibiotic", "mic", "resistant"])
    df.to_csv(save_path, index=False)
    print(f"Dataset salvato in {save_path}")

if __name__ == "__main__":
    generate_amr_dataset()
