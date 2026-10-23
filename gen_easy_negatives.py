import pandas as pd
from collections import defaultdict
import requests
from time import sleep
from Bio import SeqIO
from io import StringIO

# -------------------------------
# Step 1: Load phosphosite data
# -------------------------------
df = pd.read_csv("PSPA_ST.csv")

# Extract known phosphosites: (uniprot, 15-mer)
phospho_df = df[['Uniprot Primary Accession', 'SITE_+/-7_AA']].dropna()
phospho_df.columns = ['uniprot_id', 'peptide']
phospho_df['uniprot_id'] = phospho_df['uniprot_id'].str.strip().str.upper()

# Collect phosphosite peptides per UniProt ID
phosphosite_by_uniprot = defaultdict(set)
for _, row in phospho_df.iterrows():
    phosphosite_by_uniprot[row['uniprot_id']].add(row['peptide'])

unique_ids = list(phosphosite_by_uniprot.keys())

# -------------------------------
# Step 2: Batch fetch FASTA from UniProt
# -------------------------------
def fetch_uniprot_batch_fasta(uniprot_ids, chunk_size=500):
    fasta_seqs = {}
    url = "https://rest.uniprot.org/uniprotkb/stream"

    for i in range(0, len(uniprot_ids), chunk_size):
        chunk = uniprot_ids[i:i + chunk_size]
        query = f"(accession:{' OR accession:'.join(chunk)})"
        params = {"format": "fasta", "query": query}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            fasta_text = response.text
            for record in SeqIO.parse(StringIO(fasta_text), "fasta"):
                fasta_seqs[record.id.split("|")[1]] = str(record.seq)  # UniProt ID is in 2nd field
            print(f"✅ Fetched {len(chunk)} sequences")
        else:
            print(f"❌ Failed for chunk {i}-{i+chunk_size}, status {response.status_code}")

        sleep(1)  # avoid rate limiting

    return fasta_seqs

uniprot_seqs = fetch_uniprot_batch_fasta(unique_ids)
print("✅ Finished fetching UniProt sequences.")

# -------------------------------
# Step 3: Extract easy negatives
# -------------------------------
easy_negatives = []

for uid, seq in uniprot_seqs.items():
    known_peptides = phosphosite_by_uniprot[uid]

    for i in range(7, len(seq) - 7):
        aa = seq[i]
        if aa in {'S', 'T'}:
            window = seq[i-7:i+8]
            if 'X' in window or len(window) != 15:
                continue  # skip invalid windows
            if window not in known_peptides:
                easy_negatives.append((uid, i + 1, aa, window))  # +1 to convert to 1-based position

print(f"✅ Found {len(easy_negatives)} easy negative examples.")

# -------------------------------
# Step 4: Save to CSV
# -------------------------------
neg_df = pd.DataFrame(easy_negatives, columns=["uniprot_id", "position", "residue", "15mer"])
neg_df.to_csv("easy_negatives.csv", index=False)
print("📁 Saved to easy_negatives.csv")
