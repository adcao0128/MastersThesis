from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold, add_peptide_bonds
from functools import partial
new_edge_funcs = {"edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=5, threshold=10.), add_peptide_bonds]}
config = ProteinGraphConfig(**new_edge_funcs)
def is_file_openable(path):
    try:
        with open(path, 'r') as f:
            graph = construct_graph(config=config, path=path)
            pass  # Successfully opened and closed
        return True
    except Exception as e:
        print(f"Failed to open {path}: {e}")
        return False

if is_file_openable('./alphafold_pdb_files/O43526.pdb'):
    print("File is openable")
else:
    print("File is not openable")