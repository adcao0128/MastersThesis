import pandas as pd

# === 1. Parse CD-HIT kinase cluster assignments ===
def parse_cd_hit_clusters(clstr_file):
    cluster_map = {}
    with open(clstr_file) as f:
        cluster_id = -1
        for line in f:
            if line.startswith(">Cluster"):
                cluster_id += 1
            else:
                parts = line.strip().split('>')
                if len(parts) > 1:
                    full = parts[1].split('...')[0]
                    cluster_map[full] = cluster_id
    return cluster_map

# === 2. Map kinase gene names to CD-HIT FASTA names ===
def parse_fasta_kinase_names(fasta_file):
    name_map = {}
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                parts = line.strip().split()
                cdhit_name = parts[0][1:]  # Remove '>'
                for part in parts:
                    if part.startswith("gene="):
                        gene_name = part.split("=")[1]
                        name_map[gene_name] = cdhit_name
    return name_map

# === 3. Parse peptide cluster mapping ===
def map_site_to_cluster(peptide_cluster_file):
    df = pd.read_csv(peptide_cluster_file)
    return {
        f"{row['Uniprot Primary Accession']}_{row['SITE_+/-7_AA']}": f"{row['phos_res'].upper()}_{row['cluster_id']}"
        for _, row in df.iterrows()
    }

def assign_clusters_to_negatives(pos_file, kinase_name_map, kinase_cluster_map, site_cluster_map, kinase_uniprot_map):
    neg_df = pd.read_csv(pos_file, names=['site', 'kinase', 'percentile'])
    print(f"Loaded {len(neg_df)} negative examples.")
    print(neg_df.head())
    # Kinase → CD-HIT name → cluster
    neg_df['cdhit_kinase'] = neg_df['kinase'].map(kinase_name_map)
    neg_df['kinase_cluster'] = neg_df['cdhit_kinase'].map(kinase_cluster_map)

    # Extract Uniprot accession and 15-mer window from `site`
    neg_df[['site_acc', 'site_window']] = neg_df['site'].str.split('_', n=1, expand=True)
    neg_df['site_key'] = neg_df['site_acc'] + "_" + neg_df['site_window']

    # Site → peptide cluster
    neg_df['site_cluster'] = neg_df['site_key'].map(site_cluster_map)

    # Kinase → UniProt ID
    neg_df['kinase_uniprot_id'] = neg_df['kinase'].map(kinase_uniprot_map)

    # Drop incomplete rows
    neg_df.dropna(subset=['kinase_cluster', 'site_cluster'], inplace=True)

    # Combine into unique cluster pair
    neg_df['cluster_pair'] = list(zip(neg_df['kinase_cluster'], neg_df['site_cluster']))

    # Drop duplicates on cluster_pair
    neg_df_unique = neg_df.drop_duplicates(subset=['cluster_pair'])

    return neg_df_unique


# === 5. Run everything ===
if __name__ == "__main__":
    # File paths (adjust as needed)
    neg_file = "negative_kinase_site_pairs.csv"
    kinase_fasta = "filtered70"
    kinase_clstr = "filtered70.clstr"
    peptide_clusters = "PSPA_ST_with_clusters.csv"
    kinase_map_file = "kinase_mapping.csv"  # <-- Add this file

    # Load mappings
    kinase_name_map = parse_fasta_kinase_names(kinase_fasta)
    kinase_cluster_map = parse_cd_hit_clusters(kinase_clstr)
    site_cluster_map = map_site_to_cluster(peptide_clusters)

    # Load kinase symbol → UniProt ID mapping
    kinase_mapping_df = pd.read_csv(kinase_map_file)
    kinase_uniprot_map = dict(zip(kinase_mapping_df['filter_symbol'], kinase_mapping_df['uniprot_id']))

    # Run clustering assignment
    final_df = assign_clusters_to_negatives(
        neg_file, kinase_name_map, kinase_cluster_map, site_cluster_map, kinase_uniprot_map
    )

    # Save result
    final_df.to_csv("unique_negative_cluster_pairs.csv", index=False)
    print(f"Saved {len(final_df)} unique negative cluster pairs.")
