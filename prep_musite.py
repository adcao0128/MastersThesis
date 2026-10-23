# from Bio import SeqIO
# import pandas as pd

# fasta_file = "./MusiteDatasets/trainning_proteins_MAPK.fasta"
# window_size = 7  # Number before and after
# results = []

# for record in SeqIO.parse(fasta_file, "fasta"):
#     # Extract UniProt ID from header
#     parts = record.id.split("|")
#     uniprot_id = parts[1] if len(parts) > 1 else record.id

#     seq_str = str(record.seq)
#     i = 0
#     while i < len(seq_str):
#         aa = seq_str[i]

#         if aa in ["S", "T"]:
#             # Positive if followed by '#'
#             if i + 1 < len(seq_str) and seq_str[i+1] == "#":
#                 label = 1
#                 # Remove '#' for window
#                 clean_seq = seq_str[:i+1] + seq_str[i+2:]
#                 site_pos = i
#                 i += 1  # Skip '#'
#             else:
#                 label = 0
#                 clean_seq = seq_str
#                 site_pos = i

#             # Extract window
#             start = max(0, site_pos - window_size)
#             end = min(len(clean_seq), site_pos + window_size + 1)
#             window_seq = clean_seq[start:end]

#             # Pad to fixed length if needed
#             if len(window_seq) < (2 * window_size + 1):
#                 left_pad = window_size - (site_pos - start)
#                 right_pad = window_size - (end - site_pos - 1)
#                 window_seq = ("_" * left_pad) + window_seq + ("_" * right_pad)

#             results.append([uniprot_id, window_seq, label])

#         i += 1

# # Save to CSV
# df = pd.DataFrame(results, columns=["UniprotID", "Window", "Label"])
# df.to_csv("protein_sites_windows.csv", index=False)

# print("CSV saved as protein_sites_windows.csv")

# for record in SeqIO.parse(fasta_file, "fasta"):
#     parts = record.id.split("|")
#     uniprot_id = parts[1] if len(parts) > 1 else record.id
#     seq_str = str(record.seq)

#     i = 0
#     while i < len(seq_str):
#         aa = seq_str[i]
#         if aa in ["S", "T"] and i + 1 < len(seq_str) and seq_str[i+1] == "#":
#             # Remove '#' for clean sequence
#             clean_seq = seq_str[:i+1] + seq_str[i+2:]
#             site_pos = i
#             i += 1  # Skip the '#'

#             # Extract window
#             start = max(0, site_pos - window_size)
#             end = min(len(clean_seq), site_pos + window_size + 1)
#             window_seq = clean_seq[start:end]

#             # Pad if needed
#             if len(window_seq) < (2 * window_size + 1):
#                 left_pad = window_size - (site_pos - start)
#                 right_pad = window_size - (end - site_pos - 1)
#                 window_seq = ("-" * left_pad) + window_seq + ("-" * right_pad)

#             positive_sites.add((uniprot_id, window_seq.upper()))  # <-- UPPERCASE here
#         i += 1

# print(f"Positive sites from FASTA: {len(positive_sites)}")


# # ==== STEP 2: Load Kinase_Substrate_Dataset.txt ====
# psp = pd.read_csv("Kinase_Substrate_Dataset.txt", sep="\t")
# psp_subset = psp[["KIN_ACC_ID", "SUB_ACC_ID", "SITE_+/-7_AA"]]

# # Make set of substrate ID + window (uppercased)
# psp_sites = set(
#     zip(
#         psp_subset["SUB_ACC_ID"],
#         psp_subset["SITE_+/-7_AA"].str.upper()  # <-- UPPERCASE here
#     )
# )

# print(f"Positive sites from PSP dataset: {len(psp_sites)}")


# # ==== STEP 3: Intersection ====
# overlap = positive_sites & psp_sites
# print(f"Overlap count: {len(overlap)}")

# # Optional: save matches
# overlap_df = pd.DataFrame(overlap, columns=["UniprotID", "Window"])

# print(overlap_df.info())

from Bio import SeqIO
import pandas as pd

# ==== STEP 1: Extract positives from FASTA ====
fasta_file = "./MusiteDatasets/trainning_proteins_MAPK.fasta"
window_size = 7
positive_sites = set()

for record in SeqIO.parse(fasta_file, "fasta"):
    parts = record.id.split("|")
    uniprot_id = parts[1] if len(parts) > 1 else record.id
    seq_str = str(record.seq)

    i = 0
    while i < len(seq_str):
        aa = seq_str[i]
        if aa in ["S", "T"] and i + 1 < len(seq_str) and seq_str[i+1] == "#":
            # Remove '#' for clean sequence
            clean_seq = seq_str[:i+1] + seq_str[i+2:]
            site_pos = i
            i += 1  # Skip the '#'

            # Extract window
            start = max(0, site_pos - window_size)
            end = min(len(clean_seq), site_pos + window_size + 1)
            window_seq = clean_seq[start:end]

            # Pad if needed
            if len(window_seq) < (2 * window_size + 1):
                left_pad = window_size - (site_pos - start)
                right_pad = window_size - (end - site_pos - 1)
                window_seq = ("_" * left_pad) + window_seq + ("_" * right_pad)

            positive_sites.add((uniprot_id, window_seq.upper()))  # <-- UPPERCASE here
        i += 1

print(f"Positive sites from FASTA: {len(positive_sites)}")


# ==== STEP 2: Load Kinase_Substrate_Dataset.txt ====
psp = pd.read_csv("Kinase_Substrate_Dataset.txt", sep="\t")
psp_subset = psp[["KIN_ACC_ID", "SUB_ACC_ID", "SITE_+/-7_AA"]]

# Make set of substrate ID + window (uppercased)
psp_sites = set(
    zip(
        psp_subset["SUB_ACC_ID"],
        psp_subset["SITE_+/-7_AA"].str.upper()  # <-- UPPERCASE here
    )
)

print(f"Positive sites from PSP dataset: {len(psp_sites)}")


# ==== STEP 3: Intersection ====
overlap = positive_sites & psp_sites
print(f"Overlap count: {len(overlap)}")

# Optional: save matches
overlap_df = pd.DataFrame(overlap, columns=["UniprotID", "Window"])
print(overlap_df.info())

fasta_only = positive_sites - psp_sites
print(f"FASTA-only positives: {len(fasta_only)}")

# Save to CSV
fasta_only_df = pd.DataFrame(fasta_only, columns=["UniprotID", "Window"])
fasta_only_df.to_csv("fasta_only_positive_sites.csv", index=False)
print("Saved FASTA-only positives to fasta_only_positive_sites.csv")

print(fasta_only_df.info())
