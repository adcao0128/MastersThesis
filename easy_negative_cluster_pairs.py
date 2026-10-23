import pandas as pd
import random

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

# === 3. Generate Easy Negatives ===
def generate_easy_negative_pairs(peptide_file, kinase_name_map, kinase_cluster_map, max_pairs=2_000_000):
    # Load peptides
    pep_df = pd.read_csv(peptide_file)

    kinase_map_df = pd.read_csv("kinase_mapping.csv")
    kinase_map_df['filter_symbol'] = kinase_map_df['filter_symbol'].str.upper()
    symbol_to_uniprot = dict(zip(kinase_map_df['filter_symbol'], kinase_map_df['uniprot_id']))

    # Compute site_cluster ID
    pep_df['site_cluster'] = pep_df['residue'].str.upper() + "_" + pep_df['cluster_id'].astype(str)
    pep_df['site_key'] = pep_df['uniprot_id'] + "_" + pep_df['15mer']

    # Keep only one peptide per site_cluster
    pep_df_unique = pep_df.drop_duplicates(subset=['site_cluster'])

    # Create peptide pool
    peptide_pool = pep_df_unique[['uniprot_id', '15mer', 'residue', 'site_cluster', 'site_key']].reset_index(drop=True)

    # Create kinase pool
    kinase_df = pd.DataFrame([
        {
            'kinase': k,
            'cdhit_name': cdhit,
            'kinase_cluster': kinase_cluster_map.get(cdhit),
            'kinase_uniprot_id': symbol_to_uniprot.get(k.upper())  # map UniProt ID here
        }
        for k, cdhit in kinase_name_map.items()
        if kinase_cluster_map.get(cdhit) is not None and k.upper() in symbol_to_uniprot
    ])

    # Drop duplicates and clean
    kinase_df = kinase_df.drop_duplicates(subset=['kinase_cluster']).reset_index(drop=True)

    # Shuffle both
    peptide_pool = peptide_pool.sample(frac=1, random_state=42).reset_index(drop=True)
    kinase_df = kinase_df.sample(frac=1, random_state=42).reset_index(drop=True)

    used_pairs = set()
    rows = []

    for _, kin in kinase_df.iterrows():
        for _, pep in peptide_pool.iterrows():
            pair = (kin['kinase_cluster'], pep['site_cluster'])
            if pair not in used_pairs:
                used_pairs.add(pair)
                rows.append({
                    'kinase': kin['kinase'],
                    'kinase_uniprot_id': kin['kinase_uniprot_id'],  # now populated
                    'kinase_cluster': kin['kinase_cluster'],
                    'site_uniprot_id': pep['uniprot_id'],
                    'residue': pep['residue'],
                    '15mer': pep['15mer'],
                    'site_cluster': pep['site_cluster']
                })
                if len(rows) >= max_pairs:
                    print(f"Reached max pair limit: {max_pairs}")
                    break
        if len(rows) >= max_pairs:
            break
        if len(rows) % 1_000_000:
            print(len(rows))


    # Cap at max_pairs
    total = len(rows)
    if total > max_pairs:
        print(f"Generated {total} unique cluster pairs, truncating to {max_pairs}.")
        rows = rows[:max_pairs]
    else:
        print(f"Generated {total} unique cluster pairs. No truncation needed.")

    df_easy = pd.DataFrame(rows)
    return df_easy

# === 4. Run Everything ===
if __name__ == "__main__":
    peptide_file = "easy_negatives_with_clusters.csv"
    kinase_fasta = "filtered70"
    kinase_clstr = "filtered70.clstr"

    kinase_name_map = parse_fasta_kinase_names(kinase_fasta)
    kinase_cluster_map = parse_cd_hit_clusters(kinase_clstr)

    df_easy = generate_easy_negative_pairs(
        peptide_file, kinase_name_map, kinase_cluster_map, max_pairs=20_000_000
    )

    df_easy.to_csv("easy_negative_pairs_unique_clusters.csv", index=False)
    print(f"Saved {len(df_easy)} easy negative peptide–kinase pairs with unique cluster combinations.")
