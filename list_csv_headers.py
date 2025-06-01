import os

def list_csv_headers(base_dir="data/DATI REALI"):
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    print(f"Trovati {len(csv_files)} file CSV.\n")
    for path in csv_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
            print(f"{path}:\n  {header}\n")
        except Exception as e:
            print(f"{path}:\n  Errore lettura header: {e}\n")

if __name__ == "__main__":
    list_csv_headers()
