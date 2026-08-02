import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define paths
source_dir = "../../data/splits"         # Where your flat 2-way files currently sit
track_b_dir = "../../data/splits"        # Track B expects: data/splits/davis/cold_target/
moltrans_dir = "./dataset"               # MolTrans expects: ./dataset/davis_cold_target/

datasets = ["davis", "kiba"]
split_types = ["random", "cold_drug", "cold_target", "cold_pair"]

def process_and_save(df, dataset, split, split_fold):
    # Rename columns for baseline compatibility
    df = df.rename(columns={
        "Drug": "SMILES",
        "Target": "Target Sequence",
        "Y": "Label"
    })
    
    # Binarize the labels for classification
    if dataset == "davis":
        df["Label"] = (df["Label"] >= 7.0).astype(int)
    elif dataset == "kiba":
        df["Label"] = (df["Label"] >= 12.1).astype(int)
    
    # 1. Save for Track B (requires 'valid.csv' nested in subdirectories)
    track_b_folder = os.path.join(track_b_dir, dataset, split)
    os.makedirs(track_b_folder, exist_ok=True)
    track_b_file = "valid.csv" if split_fold == "valid" else f"{split_fold}.csv"
    df.to_csv(os.path.join(track_b_folder, track_b_file), index=False)
    
    # 2. Save for MolTrans (requires 'val.csv' in dataset folder)
    moltrans_folder = os.path.join(moltrans_dir, f"{dataset}_{split}")
    os.makedirs(moltrans_folder, exist_ok=True)
    moltrans_file = "val.csv" if split_fold == "valid" else f"{split_fold}.csv"
    df.to_csv(os.path.join(moltrans_folder, moltrans_file), index=False)

for dataset in datasets:
    for split in split_types:
        train_source = os.path.join(source_dir, f"{dataset}_{split}_train.csv")
        test_source = os.path.join(source_dir, f"{dataset}_{split}_test.csv")
        
        if os.path.exists(train_source) and os.path.exists(test_source):
            # Load the 2-way splits
            train_full_df = pd.read_csv(train_source)
            test_df = pd.read_csv(test_source)
            
            # --- FIX: Create the missing validation set ---
            # Splitting 10% of the training data to act as the validation set
            train_df, valid_df = train_test_split(train_full_df, test_size=0.1, random_state=42)
            
            # Process and save all 3 splits to BOTH required locations
            process_and_save(train_df, dataset, split, "train")
            process_and_save(valid_df, dataset, split, "valid")
            process_and_save(test_df, dataset, split, "test")
            
            print(f"Processed 3-way splits for: {dataset} - {split}")
        else:
            print(f"Warning: Could not find flat files for {dataset}_{split}")

print("\nAll custom splits successfully formatted, binarized, and saved for both pipelines!")