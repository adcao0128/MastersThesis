import pandas as pd

# Load both CSVs
full_df = pd.read_csv("scraped_kinase_mapping.csv")
ref_df = pd.read_csv("kinase_mapping.csv")

# Filter full_df to only include rows with uniprot_id in the reference set
filtered_df = full_df[full_df["uniprot_id"].isin(ref_df["uniprot_id"])]

# Save filtered results
filtered_df.to_csv("filtered_kinase_mapping.csv", index=False)

print(f"Filtered down to {len(filtered_df)} entries (from {len(full_df)})")
