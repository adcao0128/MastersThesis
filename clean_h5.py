import pandas as pd
import h5py

# === Paths ===
csv_path = "all_labeled_kinase_peptide_pairs_balanced.csv"
h5_path = "../../../../scratch/pbsjobs/job.2634651.hpc/protein_data.h5"
output_csv = "pspa_filtered_balanced_dataset.csv"

# === Load CSV ===
df = pd.read_csv(csv_path)

# === Load UniProt IDs from HDF5 ===
with h5py.File(h5_path, "r") as hf:
    h5_keys = set(hf.keys())  # Set of all UniProt IDs in the HDF5

# === Filter out rows where kinase or substrate is missing in HDF5 ===
before = len(df)
df_filtered = df[df["KIN_ACC_ID"].isin(h5_keys) & df["SUB_ACC_ID"].isin(h5_keys)]
after = len(df_filtered)

print(f"✅ Filtered {before - after} rows. {after} remain.")

# === Save cleaned CSV ===
df_filtered.to_csv(output_csv, index=False)
print(f"✅ Saved cleaned CSV to {output_csv}")
