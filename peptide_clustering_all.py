import pandas as pd

def fast_hamming_dist(a, b, max_dist=5):
    mismatches = 0
    for c1, c2 in zip(a, b):
        if c1 != c2:
            mismatches += 1
            if mismatches > max_dist:
                return mismatches  # early exit
    return mismatches

def greedy_cluster(sequences, max_dist=5):
    clusters = []
    representatives = []  # first sequence of each cluster

    for i, seq in enumerate(sequences):
        if i % 10000 == 0:
            print(f"Clustering peptide {i}/{len(sequences)}")

        # Try to assign to an existing cluster by comparing to representatives only
        assigned = False
        for cluster_idx, rep_seq in enumerate(representatives):
            if fast_hamming_dist(seq, rep_seq, max_dist) <= max_dist:
                clusters[cluster_idx].append(i)
                assigned = True
                break

        # If not assigned, start a new cluster with this sequence as representative
        if not assigned:
            clusters.append([i])
            representatives.append(seq)

    return clusters

# Load PSPA data (positives)
df = pd.read_csv("PSPA_ST.csv")

# Peptides and central residue (positives)
pos_peptides = df["SITE_+/-7_AA"].dropna().astype(str)
pos_central_res = pos_peptides.str[7]
pos_central_11mers = pos_peptides.str[2:13]

# Your easy negatives peptides, loaded from CSV (adjust path)
neg_df = pd.read_csv("easy_negatives.csv")
neg_peptides = neg_df["15mer"].astype(str).str[2:13]  # slice 11-mer from 15-mer
neg_central_res = neg_peptides.str[5]

# Combine positive and negative peptides & residues
combined_11mers = pd.concat([pos_central_11mers, neg_peptides], ignore_index=True)
combined_central_res = pd.concat([pos_central_res, neg_central_res], ignore_index=True)

# Separate combined peptides by residue type
ser_indices = combined_central_res[combined_central_res == "S"].index
thr_indices = combined_central_res[combined_central_res == "T"].index

ser_peptides = combined_11mers.loc[ser_indices].tolist()
thr_peptides = combined_11mers.loc[thr_indices].tolist()

print(len(combined_11mers))
print(len(ser_peptides))
print(len(thr_peptides))

# Cluster serine and threonine peptides
print("Clustering serine sites...")
ser_clusters = greedy_cluster(ser_peptides, max_dist=5)
print(f"✅ Serine clusters: {len(ser_clusters)}")

print("Clustering threonine sites...")
thr_clusters = greedy_cluster(thr_peptides, max_dist=5)
print(f"✅ Threonine clusters: {len(thr_clusters)}")

# Helper to build index → cluster map
def build_cluster_map(cluster_list, indices):
    cluster_map = {}
    for cluster_id, members in enumerate(cluster_list):
        for m in members:
            cluster_map[indices[m]] = cluster_id
    return cluster_map

ser_cluster_map = build_cluster_map(ser_clusters, ser_indices.tolist())
thr_cluster_map = build_cluster_map(thr_clusters, thr_indices.tolist())

# Map cluster IDs back to all combined peptides (pos + neg)
combined_cluster_ids = pd.Series([None]*len(combined_11mers))
for idx in ser_cluster_map:
    combined_cluster_ids[idx] = ser_cluster_map[idx]
for idx in thr_cluster_map:
    combined_cluster_ids[idx] = thr_cluster_map[idx]

# Separate cluster IDs for positives and negatives
pos_cluster_ids = combined_cluster_ids.iloc[:len(pos_central_11mers)]
neg_cluster_ids = combined_cluster_ids.iloc[len(pos_central_11mers):]

# Add cluster IDs to original positive DataFrame
df["cluster_id"] = pos_cluster_ids.values

# Add cluster IDs to negatives DataFrame
neg_df["cluster_id"] = neg_cluster_ids.values

# Save both results
df.to_csv("PSPA_ST_with_clusters.csv", index=False)
neg_df.to_csv("easy_negatives_with_clusters.csv", index=False)

print("✅ Clustering complete and saved for positives and easy negatives.")
