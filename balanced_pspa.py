import pandas as pd

# === Load Data ===
df_pos = pd.read_csv("unique_positive_cluster_pairs.csv")
df_neg = pd.read_csv("unique_negative_cluster_pairs.csv")
df_easy = pd.read_csv("easy_negative_pairs_unique_clusters.csv")

# === Prepare positive examples ===
df_pos_cleaned = pd.DataFrame({
    'KIN_ACC_ID': df_pos['kinase_uniprot_id'],
    'SUB_ACC_ID': df_pos['site_acc'],
    'SITE_+/-7_AA': df_pos['site_window'],
    'true_label': 1
})

# === Prepare hard negative examples ===
df_neg_cleaned = pd.DataFrame({
    'KIN_ACC_ID': df_neg['kinase_uniprot_id'],
    'SUB_ACC_ID': df_neg['site_acc'],
    'SITE_+/-7_AA': df_neg['site_window'],
    'true_label': -1
})

# === Prepare easy negative examples ===
df_easy_cleaned = pd.DataFrame({
    'KIN_ACC_ID': df_easy['kinase_uniprot_id'].str.upper(),  # make sure uppercase UniProt IDs
    'SUB_ACC_ID': df_easy['site_uniprot_id'],
    'SITE_+/-7_AA': df_easy['15mer'],
    'true_label': 0  # use 0 label to match negative class
})

# Drop unmapped or missing KIN_ACC_ID rows
df_easy_cleaned = df_easy_cleaned.dropna(subset=['KIN_ACC_ID'])

# === Determine baseline count based on positive samples ===
n_pos = len(df_pos_cleaned)
print(f"Number of positive samples: {n_pos}")

# Sample hard negatives to match positive count (or all if fewer)
df_neg_sampled = df_neg_cleaned.sample(n=min(n_pos, len(df_neg_cleaned)), random_state=42)

# Sample easy negatives to match positive count (or all if fewer)
df_easy_sampled = df_easy_cleaned.sample(n=min(n_pos, len(df_easy_cleaned)), random_state=42)

print(f"Sampled {len(df_neg_sampled)} hard negatives and {len(df_easy_sampled)} easy negatives.")

# === Combine all balanced datasets ===
df_balanced = pd.concat([df_pos_cleaned, df_neg_sampled, df_easy_sampled], ignore_index=True)

# Shuffle combined dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# === Save balanced dataset ===
df_balanced.to_csv("all_labeled_kinase_peptide_pairs_balanced.csv", index=False)
print(f"\n✅ Balanced dataset saved with {len(df_balanced)} entries (approx. 1:1:1 ratio).")
