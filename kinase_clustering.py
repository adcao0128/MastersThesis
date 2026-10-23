from Bio import SeqIO
import requests
import pandas as pd

def fetch_fasta(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except:
        return None

kinase_accessions = pd.read_csv("reference_human_kinases.csv")[['gene', 'uniprot']]
df = pd.read_csv("PSPA_ST.csv")
percentile_cols = [col for col in df.columns if col.endswith('_percentile')]
kinases_in_df = [col.replace('_percentile', '') for col in percentile_cols]

# Define accessions to remove
to_remove = {"O14874", "Q15118", "Q16654"}

# Filter them out
kinase_accessions = kinase_accessions[~kinase_accessions.isin(to_remove)]
kinase_accessions = kinase_accessions.drop_duplicates()
print(kinase_accessions)

with open("kinases300.fasta", "w") as fasta_out:
    for acc in kinase_accessions:
        fasta = fetch_fasta(acc)
        if fasta:
            fasta_out.write(fasta)