import pandas as pd
from Bio import SeqIO

# Load missing gene names (uppercased)
with open("missing_genes.txt") as f:
    missing_genes = set(g.strip().upper() for g in f.readlines())

# Load CSV and map gene → sequence
df = pd.read_csv("reference_human_kinases.csv")
df["gene"] = df["gene"].str.upper()
gene_to_seq = dict(zip(df["gene"], df["sequence"].str.replace(" ", "").str.strip()))

# Load the full domain FASTA (where we want to find exact sequence matches)
domain_records = list(SeqIO.parse("domain.fasta", "fasta"))
seq_to_record = {str(record.seq).strip(): record for record in domain_records}

# Load the current filtered FASTA so we don't duplicate
existing_filtered = list(SeqIO.parse("filtered_by_gene.fasta", "fasta"))
existing_records = {str(r.seq) for r in existing_filtered}

# Find and rescue matching records
rescued = []
not_found = []

for gene in missing_genes:
    seq = gene_to_seq.get(gene)
    if not seq:
        not_found.append((gene, "no_sequence_in_csv"))
        continue

    seq = seq.strip().replace(" ", "")
    if seq in seq_to_record:
        if seq not in existing_records:
            rescued.append(seq_to_record[seq])
    else:
        not_found.append((gene, "sequence_not_found_in_fasta"))

# Save rescued entries into a new FASTA file
SeqIO.write(rescued, "rescued_missing_entries.fasta", "fasta")

print(f"✅ Rescued {len(rescued)} missing genes into 'rescued_missing_entries.fasta'")
if not_found:
    print(f"❌ {len(not_found)} could not be rescued:")
    for gene, reason in not_found:
        print(f"  - {gene}: {reason}")
