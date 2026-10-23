import pandas as pd
import os
import csv
from Bio import PDB
from Bio.PDB import PDBParser, PPBuilder


# Configuration
PDB_FOLDER = "./alphafold_pdb_files"
WINDOW = 7
PAD_CHAR = "_"

# Load dataset
df = pd.read_csv("GPS_clean.csv")

def clean_accession(acc):
    """Remove isoform suffix from UniProt ID (e.g., Q5NUL3-2 -> Q5NUL3)"""
    if pd.isna(acc):
        return None
    return acc.split("-")[0]

def extract_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    ppb = PPBuilder()
    full_seq = ""
    for pp in ppb.build_peptides(structure):
        # pp.get_sequence() returns a Bio.Seq.Seq object
        full_seq += str(pp.get_sequence())
    return full_seq

def get_peptide(seq, pos_1based, window=WINDOW, pad_char=PAD_CHAR):
    """Extract PSP(window,window) peptide centered at pos (1-based)."""
    # Convert 1-based position to 0-based index
    center_idx = pos_1based - 1
    start = center_idx - window
    end = center_idx + window + 1  # +1 because slice end is exclusive
    peptide = "".join(seq[i] if 0 <= i < len(seq) else pad_char for i in range(start, end))
    return peptide

def generate_peptides(df, pdb_folder=PDB_FOLDER, window=WINDOW, pad_char=PAD_CHAR):
    """Generator yielding positive and negative peptides row by row."""
    sequence_cache = {}

    for idx, row in df.iterrows():
        sub_id_raw = row["SUB_ACC_ID"]
        kin_id_raw = row["KIN_ACC_ID"]
        pos = int(row["Position"])
        code = row["Code"]

        # Strip isoforms
        sub_id = clean_accession(sub_id_raw)
        kin_id = clean_accession(kin_id_raw)
        if sub_id is None or kin_id is None:
            continue

        # Fetch or get cached sequence from PDB
        if sub_id not in sequence_cache:
            pdb_file = os.path.join(pdb_folder, f"{sub_id}.pdb")
            if not os.path.exists(pdb_file):
                print(f"PDB not found for {sub_id}")
                continue
            seq = extract_sequence_from_pdb(pdb_file)
            if seq:
                sequence_cache[sub_id] = seq
            else:
                continue
        else:
            seq = sequence_cache[sub_id]

        # --- Positive peptide ---
        pos_peptide = get_peptide(seq, pos, window, pad_char)

        # Sanity check: center matches Code
        center_residue = pos_peptide[window]  # center of peptide
        if center_residue != code:
            print(f"Warning: center mismatch for {sub_id} pos {pos}. Expected {code}, found {center_residue}")
            continue

        yield {"KIN_ACC_ID": kin_id,
               "SUB_ACC_ID": sub_id,
               "SITE_+/-7_AA": pos_peptide,
               "Label": 1}

        # --- Negative peptides: same residue type ---
        for i, res in enumerate(seq):
            if res == code and (i + 1) != pos:  # skip positive site
                neg_peptide = get_peptide(seq, i + 1, window, pad_char)
                # Check center matches residue
                if neg_peptide[window] != code:
                    print(f"Warning: negative center mismatch for {sub_id} pos {i+1}")
                    continue
                yield {"KIN_ACC_ID": kin_id,
                       "SUB_ACC_ID": sub_id,
                       "SITE_+/-7_AA": neg_peptide,
                       "Label": 0}

# Write peptides to CSV
output_file = "GPS_psp_7_7_alphafold_checked.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["KIN_ACC_ID", "SUB_ACC_ID", "SITE_+/-7_AA", "Label"])
    writer.writeheader()
    for peptide_row in generate_peptides(df):
        writer.writerow(peptide_row)

print(f"Peptide extraction complete. Saved to {output_file}")
