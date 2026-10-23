import pandas as pd
from collections import defaultdict

def hamming_dist(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))

def greedy_cluster(sequences, max_dist=5):
    clusters = []
    assigned = set()

    for i, seq in enumerate(sequences):
        if i % 10000 == 0:
            print(i)
        if i in assigned:
            continue

        cluster = [i]
        assigned.add(i)

        for j in range(i + 1, len(sequences)):
            if j not in assigned and hamming_dist(seq, sequences[j]) <= max_dist:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)
    return clusters

# Load PSPA data
df = pd.read_csv("PSPA_ST.csv")

# Get 15-mer peptides
peptides = df["SITE_+/-7_AA"].dropna().astype(str)

# Extract central residue and central 11-mer (positions 2–12)
central_res = peptides.str[7]
central_11mers = peptides.str[2:13]

# Separate serine and threonine
ser_indices = central_res[central_res == "S"].index
thr_indices = central_res[central_res == "T"].index

ser_peptides = central_11mers.loc[ser_indices].tolist()
thr_peptides = central_11mers.loc[thr_indices].tolist()

# Cluster
print("Clustering serine sites...")
ser_clusters = greedy_cluster(ser_peptides, max_dist=5)
print(f"✅ Serine clusters: {len(ser_clusters)}")

print("Clustering threonine sites...")
thr_clusters = greedy_cluster(thr_peptides, max_dist=5)
print(f"✅ Threonine clusters: {len(thr_clusters)}")


# Helper to generate mapping: index in original list → cluster ID
def build_cluster_map(cluster_list, original_indices):
    cluster_map = {}
    for cluster_id, members in enumerate(cluster_list):
        for i in members:
            cluster_map[original_indices[i]] = cluster_id
    return cluster_map

# Map cluster IDs
ser_cluster_map = build_cluster_map(ser_clusters, ser_indices.tolist())
thr_cluster_map = build_cluster_map(thr_clusters, thr_indices.tolist())

df["cluster_id"] = df.index.map(lambda i: ser_cluster_map.get(i) if i in ser_cluster_map
                                else thr_cluster_map.get(i) if i in thr_cluster_map
                                else None)

# Save with clusters
df.to_csv("PSPA_ST_with_clusters.csv", index=False)
