import pandas as pd

# === Load Data ===
df_pos = pd.read_csv("unique_positive_cluster_pairs.csv")
df_neg = pd.read_csv("unique_negative_cluster_pairs.csv")
df_easy = pd.read_csv("easy_negative_pairs_unique_clusters.csv")

# === Positive Examples ===
df_pos_cleaned = pd.DataFrame({
    'KIN_ACC_ID': df_pos['kinase_uniprot_id'],
    'SUB_ACC_ID': df_pos['site_acc'],
    'SITE_+/-7_AA': df_pos['site_window'],
    'true_label': 1
})

# === Hard Negative Examples ===
df_neg_cleaned = pd.DataFrame({
    'KIN_ACC_ID': df_neg['kinase_uniprot_id'],
    'SUB_ACC_ID': df_neg['site_acc'],
    'SITE_+/-7_AA': df_neg['site_window'],
    'true_label': 0
})

# === Easy Negative Examples ===
df_easy_cleaned = pd.DataFrame({
    'KIN_ACC_ID': df_easy['kinase_uniprot_id'].str.upper(),           # Initially gene symbols
    'SUB_ACC_ID': df_easy['site_uniprot_id'],
    'SITE_+/-7_AA': df_easy['15mer'],
    'true_label': -1
})

# Drop unmapped rows
df_easy_cleaned = df_easy_cleaned.dropna(subset=['KIN_ACC_ID'])

# === Combine All ===
df_all = pd.concat([df_pos_cleaned, df_neg_cleaned, df_easy_cleaned], ignore_index=True)

# === Save Final Dataset ===
df_all.to_csv("all_labeled_kinase_peptide_pairs.csv", index=False)
print(f"\n✅ Final combined dataset saved with {len(df_all)} entries.")
