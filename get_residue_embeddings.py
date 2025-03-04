import logging
import multiprocessing as mp
import os
import shutil
import re
import tempfile
import warnings
from io import TextIOWrapper
from pathlib import Path
import argparse
import torch
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
from Bio.PDB.Polypeptide import is_aa

from esm import FastaBatchedDataset, pretrained

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

if not log.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        " %(asctime)s %(module)s:%(lineno)d %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

ESM_MODEL = "esm2_t33_650M_UR50D"

TOKS_PER_BATCH = 4096
REPR_LAYERS = [0, 32, 33]
TRUNCATION_SEQ_LENGTH = 2500
INCLUDE = ["mean", "per_tok"]
BATCH_SIZE = 8
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


print(torch.__version__)

def pdb_to_fasta(pdb_file_path: Path, main_fasta_fh: TextIOWrapper) -> None:
    """Convert a PDB file with a single protein to a FASTA file."""
    log.info(f"Reading sequence of PDB {pdb_file_path.name}")
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file_path)

    # Ensure there is at least one chain in the structure
    chains = list(structure[0].get_chains())
    if not chains:
        raise ValueError(f"No chains found in PDB file {pdb_file_path.name}")

    # Get the first chain (assuming only one protein chain in the PDB file)
    chain = chains[0]  # Select the first chain
    sequence = ""

    # Get the sequence of the chain
    for residue in chain:
        if not is_aa(residue.get_resname()):
            continue
        sequence += residue.get_resname()

    # Write the sequence to a FASTA file
    root = re.findall(r"(.*).pdb", pdb_file_path.name)[0]
    main_fasta_fh.write(f">{root}\n{sequence}\n")

def get_embedding(fasta_file: Path, output_dir: Path) -> None:
    """
    Get the embedding of a protein sequence.

    Adapted from: <https://github.com/facebookresearch/esm/blob/d7b3331f41442ed4ffde70cb95bdd48cabcec2e9/scripts/extract.py#L63>
    """
    log.info("Generating embedding for protein sequence.")
    model, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        log.info("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(  # type: ignore
        dataset,
        collate_fn=alphabet.get_batch_converter(TRUNCATION_SEQ_LENGTH),
        batch_sampler=batches,
    )
    log.info(f"Read {fasta_file} with {len(dataset)} sequences")

    output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in INCLUDE

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in REPR_LAYERS)  # type: ignore
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS]  # type: ignore

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            log.info(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                output_file = output_dir / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(TRUNCATION_SEQ_LENGTH, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in INCLUDE:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in INCLUDE:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }

                if return_contacts:
                    result["contacts"] = contacts[i, :truncate_len, :truncate_len].clone()  # type: ignore

                torch.save(
                    result, output_file,
                )

def process_all_pdb_files_in_folder(pdb_folder_path: Path, output_dir: Path) -> None:
    """Process all PDB files in the folder and generate embeddings for all sequences."""
    # Create a single FASTA file to store all sequences
    combined_fasta_file = output_dir / "combined_sequences.fasta"
    
    with open(combined_fasta_file, "w") as main_fasta_fh:
        # Process each PDB file
        for pdb_file_path in pdb_folder_path.glob("*.pdb"):
            # Convert PDB to FASTA and append to the combined FASTA file
            pdb_to_fasta(pdb_file_path, main_fasta_fh)

    log.info("Generating embeddings from the combined FASTA file.")
    get_embedding(combined_fasta_file, output_dir)

if __name__ == "__main__":
    pdb_folder_path = Path("./alphafold_pdb_files")
    output_dir = Path("./residue_embeddings")
    log.info("Starting processing")
    process_all_pdb_files_in_folder(pdb_folder_path, output_dir)
    #log.info("Generating embeddings from the combined FASTA file.")
    #combined_fasta_file = output_dir / "combined_sequences.fasta"
    # get_embedding(combined_fasta_file, output_dir)


