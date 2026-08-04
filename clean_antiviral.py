import pandas as pd
import os
import numpy as np
import hashlib

print("Cleaning raw antiviral BindingDB data...")
df = pd.read_csv("data/raw/bindingdb_antiviral.csv", on_bad_lines='skip', low_memory=False)

def find_col(keywords):
    for col in df.columns:
        if all(k.lower() in col.lower() for k in keywords):
            return col
    return None

uniprot_col = find_col(['uniprot', 'primary id']) or find_col(['uniprot'])
name_col = find_col(['target', 'name'])
seq_col = find_col(['target', 'sequence'])
smiles_col = find_col(['ligand', 'smiles'])

# Find ALL affinity metrics
ic50_col = find_col(['ic50', '(nm)']) or find_col(['ic50'])
ki_col = find_col(['ki', '(nm)']) or find_col(['ki'])
kd_col = find_col(['kd', '(nm)']) or find_col(['kd'])
ec50_col = find_col(['ec50', '(nm)']) or find_col(['ec50'])

# Build the clean dataframe safely (fixes the fragmentation warning)
clean_df = pd.DataFrame()

# 1. DRUGS
clean_df['Drug'] = df[smiles_col] if smiles_col else np.nan
clean_df['Drug_ID'] = clean_df['Drug']

# 2. TARGET SEQUENCE
clean_df['Target'] = df[seq_col] if seq_col else np.nan

# 3. TARGET IDs (Fallback chain: UniProt -> Name -> Sequence Hash)
target_ids = pd.Series(np.nan, index=df.index)
if uniprot_col: target_ids = target_ids.fillna(df[uniprot_col])
if name_col: target_ids = target_ids.fillna(df[name_col])

def hash_seq(seq):
    if pd.isna(seq) or str(seq).strip() == '': return np.nan
    return "SEQ_" + hashlib.md5(str(seq).encode()).hexdigest()[:8]

target_ids = target_ids.fillna(clean_df['Target'].apply(hash_seq))
clean_df['Target_ID'] = target_ids

# 4. AFFINITIES (Fallback chain: IC50 -> Ki -> Kd -> EC50)
y_vals = pd.Series(np.nan, index=df.index)
for col in [ic50_col, ki_col, kd_col, ec50_col]:
    if col and col in df.columns:
        # Replace empty spaces with NaN so fillna works properly
        clean_col = df[col].replace(r'^\s*$', np.nan, regex=True)
        y_vals = y_vals.fillna(clean_col)
clean_df['Y'] = y_vals

print(f"\n[*] Total raw rows: {len(clean_df)}")
print("[*] Missing values BEFORE cleaning:")
print(clean_df.isna().sum().to_string())

# Drop rows missing critical data
clean_df = clean_df.dropna(subset=['Drug', 'Target', 'Target_ID', 'Y'])

if len(clean_df) > 0:
    # Clean up IDs and Affinities
    clean_df['Target_ID'] = clean_df['Target_ID'].astype(str).str.split(',').str[0].str.strip()
    clean_df['Y'] = clean_df['Y'].astype(str).str.replace('>', '').str.replace('<', '').str.replace(' ', '')
    clean_df['Y'] = pd.to_numeric(clean_df['Y'], errors='coerce')
    clean_df = clean_df.dropna(subset=['Y'])

print(f"\n[*] Rows successfully cleaned: {len(clean_df)}")

os.makedirs("data/processed", exist_ok=True)
if len(clean_df) > 0:
    clean_df.to_csv("data/processed/antiviral_clean.csv", index=False)
    print(f"[+] Success! Saved {len(clean_df)} clean records to data/processed/antiviral_clean.csv")
else:
    print("[-] Still 0 records. Check the missing values above to see what we are missing!")