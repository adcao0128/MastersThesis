import h5py
import torch
import gzip
import io

# Open the HDF5 file
with h5py.File('protein_data.h5', 'r') as f:
    pdb_data = gzip.decompress(f["A0MZ67"]["pdb"][()])
    print(pdb_data)

    pt_bytes = f["A0MZ67"]["embeddings"][:]  # Load as NumPy array

    # Convert back to BytesIO buffer
    buffer = io.BytesIO(pt_bytes.tobytes())

    # Deserialize using torch.load()
    embedding = torch.load(buffer)

    print(embedding)