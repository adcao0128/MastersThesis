from Bio import SeqIO

# Load allowed gene symbols (case-insensitive)
with open("filter_symbols.txt") as f:
    allowed_genes = set(g.strip().upper() for g in f.read().strip().split(","))

# Track matched genes
matched_genes = set()

# Input/output files
input_txt = "domain.txt"
output_fasta = "filtered_by_gene.fasta"

# Counter for saved sequences
saved_count = 0

# Parse and filter
with open(output_fasta, "w") as out_handle:
    for record in SeqIO.parse(input_txt, "fasta"):
        desc = record.description
        gene_field = next((item for item in desc.split() if item.startswith("gene=")), None)

        if gene_field:
            gene_symbol = gene_field.split("=")[1].upper()
            if gene_symbol in allowed_genes:
                SeqIO.write(record, out_handle, "fasta")
                matched_genes.add(gene_symbol)
                saved_count += 1

# Print summary
missing_genes = allowed_genes - matched_genes

print(f"\n✅ Saved {saved_count} sequences to {output_fasta}")
print(f"❌ {len(missing_genes)} gene(s) not found in FASTA:")
print(", ".join(sorted(missing_genes)))

# Optionally: save missing genes
with open("missing_genes.txt", "w") as f:
    for gene in sorted(missing_genes):
        f.write(f"{gene}\n")