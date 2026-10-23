import pandas as pd
from Bio import SeqIO
from collections import defaultdict
import re

# Load kinase mapping: gene → UniProt
df = pd.read_csv("kinase_mapping.csv")
df["filter_symbol"] = df["filter_symbol"].str.upper()
df["uniprot_id"] = df["uniprot_id"].str.upper()
gene_to_uniprot = dict(zip(df["filter_symbol"], df["uniprot_id"]))

# Parse domain FASTA and collect UniProt usage
uniprot_to_headers = defaultdict(list)

for record in SeqIO.parse("filtered_by_gene.fasta", "fasta"):
    desc = record.description

    # Extract gene name from `gene=...`
    match = re.search(r"gene=([A-Za-z0-9\-]+)", desc)
    if match:
        gene = match.group(1).upper()
        uniprot_id = gene_to_uniprot.get(gene)

        if uniprot_id:
            uniprot_to_headers[uniprot_id].append(desc)
        else:
            print(f"⚠️ Gene {gene} not found in mapping CSV")
    else:
        print(f"⚠️ No gene= found in header: {desc}")

# Find duplicated UniProt IDs
duplicates = {uid: headers for uid, headers in uniprot_to_headers.items() if len(headers) > 1}

print(f"✅ Checked {len(uniprot_to_headers)} unique UniProt IDs in FASTA")
print(f"❌ Found {len(duplicates)} UniProt ID duplicates\n")

for i, (uid, headers) in enumerate(duplicates.items(), 1):
    print(f"🔁 Duplicate #{i}: UniProt ID: {uid}")
    for header in headers:
        print(f"  - {header}")
    print()
