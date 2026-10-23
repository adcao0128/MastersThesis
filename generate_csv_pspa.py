import pandas as pd

# Load the dataset
df = pd.read_csv("PSPA_ST.csv")

# Find all columns ending in '_percentile'
percentile_cols = [col for col in df.columns if col.endswith('_percentile')]
kinases = [col.replace('_percentile', '') for col in percentile_cols]

positive_pairs = []
negative_pairs = []

for index, row in df.iterrows():
    if index % 10000 == 0:
        print(f"Finished with {index}")
    # Construct unique phosphosite identifier
    site_id = f"{row['Uniprot Primary Accession']}_{row['SITE_+/-7_AA']}"

    for kinase in kinases:
        p_col = f"{kinase}_percentile"
        if pd.notnull(row[p_col]):
            percentile = row[p_col]
            if percentile >= 90:
                positive_pairs.append((site_id, kinase, percentile))
            elif percentile <= 50:
                negative_pairs.append((site_id, kinase, percentile))

positive_pairs = list(set(positive_pairs))
negative_pairs = list(set(negative_pairs))

# Print summary
print(f"Found {len(positive_pairs)} positive kinase–site pairs")
print(f"Found {len(negative_pairs)} negative kinase–site pairs")

# Optional: Save to CSV
pd.DataFrame(positive_pairs, columns=["site", "kinase", "percentile"]).to_csv("positive_kinase_site_pairs.csv", index=False)
pd.DataFrame(negative_pairs, columns=["site", "kinase", "percentile"]).to_csv("negative_kinase_site_pairs.csv", index=False)
