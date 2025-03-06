import os
import torch
import graphein.protein as gp
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold
import networkx as nx
import pickle
from pathlib import Path
import logging
import itertools

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

if not log.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        " %(asctime)s %(module)s:%(lineno)d %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

# Paths
pdb_folder = "./alphafold_pdb_files"
embedding_folder = "./residue_embeddings"
graph_output_folder = "./protein_graphs"
os.makedirs(graph_output_folder, exist_ok=True)

# Distance threshold for graph construction
DISTANCE_THRESHOLD = 8.0
LONG_INTERACTION_THRESHOLD = 5  # Sequence separation in residues

def add_edges_with_distance_threshold(graph):
    add_distance_threshold(
        graph,
        threshold=8.0,
        long_interaction_threshold=5,
    )

config = gp.ProteinGraphConfig(
    edge_construction_functions=[add_edges_with_distance_threshold],
)

AA_DICT = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'ASX': 'B',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLX': 'Z',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
}

def load_embeddings(embedding_path):
    """Load residue-level embeddings from ESM-2 output."""
    data = torch.load(embedding_path, weights_only=False)
    return data["representations"][33]  # Residue-level embeddings from layer 33

def process_pdb_file(pdb_path):
    """Process a single PDB file to generate a residue graph with embeddings."""
    graph = construct_graph(config=config, path=str(pdb_path))

    # Load residue embeddings
    embedding_file = os.path.join(embedding_folder, f"{pdb_path.stem}.pt")
    if not os.path.exists(embedding_file):
        log.warning(f"Embedding file not found for {pdb_path.name}")
        return None

    embeddings = load_embeddings(embedding_file)

    # Add node features
    for i, (_, node_data) in enumerate(graph.nodes(data=True)):
        if i < len(embeddings):  # Ensure embeddings match the graph nodes
            node_data["feature"] = embeddings[i].numpy()
        else:
            log.warning(f"Residue mismatch in {pdb_path.name}")
            return None

    # Add edge features (e.g., distances)
    for u, v, edge_data in graph.edges(data=True):
        if "distance" in edge_data:
            edge_data["feature"] = edge_data["distance"]

    return graph

def process_all_pdb_files():
    """Generate residue graphs for all PDB files in the folder."""
    pdb_files = sorted(os.listdir(pdb_folder))
    for idx, pdb_file in enumerate(pdb_files, start=1):  # Start from 1 for human-friendly indexing
        if not pdb_file.endswith(".pdb"):
            continue

        pdb_path = os.path.join(pdb_folder, pdb_file)
        pdb_path = Path(pdb_path)
        
        log.info(f"Processing {idx}: {pdb_file}...")  # Log the index and PDB file name

        graph = process_pdb_file(pdb_path)
        if graph:
            # Save the graph
            output_file = os.path.join(graph_output_folder, f"{pdb_path.stem}-graph.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(graph, f)
            log.info(f"Saved graph for {pdb_file} to {output_file}")   

# Run the workflow
if __name__ == "__main__":
    log.info("Generating graphs from pdb files.")
    graph = construct_graph(config=config, path=str("./alphafold_pdb_files/P68101.pdb"))
    site_sequence = "MILLSELsRRRIRSI".upper()

    # k = 1  # Define the number of hops

    # # Step 1: Extract main sequence and features
    # total_sequence = ""
    # feature_vectors = []
    # node_list = []  # Maintain order of original nodes

    # for node in graph.nodes():
    #     node_id = node.split(":")[1]  # Extract amino acid ID
    #     letter = AA_DICT[node_id]  # Convert to amino acid
    #     total_sequence += letter
    #     node_list.append(node)  # Maintain order

    # # Step 2: Find start index of the subsequence
    # index = total_sequence.find(site_sequence)
    # if index == -1:
    #     print(f"Site sequence '{site_sequence}' not found in protein sequence {sub_id}.")
    #     indexed_edges = [[],[]]  # No edges

    # # Step 3: Extract the relevant nodes and features
    # selected_nodes = node_list[index : index + len(site_sequence)]

    # # Step 4: Find k-hop neighbors
    # expanded_nodes = set(selected_nodes)
    # for node in selected_nodes:
    #     neighbors = nx.single_source_shortest_path_length(graph, node, cutoff=k)
    #     expanded_nodes.update(neighbors.keys())  # Include k-hop neighbors

    # # Step 5: Extract only the valid edges from the original graph
    # expanded_edges = [(u, v) for u, v in graph.edges() if u in expanded_nodes and v in expanded_nodes]

    # # Step 6: Create a mapping from original node IDs to new indices
    # node_mapping = {old_id: new_id for new_id, old_id in enumerate(expanded_nodes)}

    # # Step 7: Create the new subgraph with updated indices
    # subgraph = nx.Graph()  # Use nx.DiGraph() if the original graph is directed
    # for old_id in expanded_nodes:
    #     new_id = node_mapping[old_id]
    #     subgraph.add_node(new_id, original_id=old_id)

    # # Step 8: Add edges with remapped indices
    # subgraph.add_edges_from([(node_mapping[u], node_mapping[v]) for u, v in expanded_edges])

    # #Debugging outputs (optional)
    # print("Subgraph nodes:", len(subgraph.nodes(data=True)))
    # print("Subgraph edges:", len(subgraph.edges()))
    # print("Graph nodes:", len(graph.nodes(data=True)))
    # print("Graph edges:", len(graph.edges()))

    #process_all_pdb_files()


# Extract sequence and features
    # total_sequence = ""
    # feature_vectors = []
    # node_list = []  # Maintain order of original nodes

    # for node in graph.nodes():
    #     node_id = node.split(":")[1]  # Extract amino acid ID
    #     letter = AA_DICT[node_id]  # Convert to amino acid
    #     total_sequence += letter
    #     node_list.append(node)  # Maintain order

    # # Find start index of the subsequence
    # index = total_sequence.find(site_sequence)
    # if index == -1:
    #     print(f"Site sequence '{site_sequence}' not found in protein sequence.")
    #     features = np.zeros((1, 1280))  # Placeholder 1-node zero vector
    #     indexed_edges = []  # No edges

    # # Extract the relevant nodes and features
    # selected_nodes = node_list[index: index + len(site_sequence)]

    # # Create a mapping from original node IDs to new indices
    # node_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}

    # # Extract and remap edges
    # selected_edges = [
    #     (node_mapping[u], node_mapping[v]) 
    #     for u, v in graph.edges() 
    #     if u in selected_nodes and v in selected_nodes
    # ]

    # # Create the subgraph with new indices
    # subgraph = nx.Graph()  # Use nx.DiGraph() if directed
    # for new_id, old_id in enumerate(zip(selected_nodes)):
    #     subgraph.add_node(new_id, original_id=old_id)  # Store old ID if needed

    # subgraph.add_edges_from(selected_edges)  # Add remapped edges

    # # Optional: Check the subgraph
    # print("Subgraph nodes:", len(subgraph.nodes()))
    # print("Subgraph edges:", len(subgraph.edges()))


    ########################

    k = 1  # Define the number of hops

    # Step 1: Extract main sequence and features
    total_sequence = ""
    feature_vectors = []
    node_list = []  # Maintain order of original nodes

    for node in graph.nodes():
        node_id = node.split(":")[1]  # Extract amino acid ID
        letter = AA_DICT[node_id]  # Convert to amino acid
        total_sequence += letter
        node_list.append(node)  # Maintain order

    # Step 2: Find start index of the subsequence
    index = total_sequence.find(site_sequence)
    if index == -1:
        print(f"Site sequence '{site_sequence}' not found in protein sequence {sub_id}.")
        indexed_edges = [[], []]  # No edges

    # Step 3: Extract the relevant nodes and features
    selected_nodes = node_list[index : index + len(site_sequence)]

    # Step 4: Find k-hop neighbors
    node_to_k_hop_neighbors = {
        node: set(nx.single_source_shortest_path_length(graph, node, cutoff=k).keys())
        for node in selected_nodes
    }

    # Step 5: Find shared k-hop neighbors and add edges between selected nodes
    subgraph = nx.Graph()  # Use nx.DiGraph() if the original graph is directed
    subgraph.add_nodes_from(selected_nodes)  # Only add selected nodes

    for node1, node2 in itertools.combinations(selected_nodes, 2):
        shared_neighbors = node_to_k_hop_neighbors[node1] & node_to_k_hop_neighbors[node2]
        if shared_neighbors:  # If they share at least one k-hop neighbor
            subgraph.add_edge(node1, node2)  # Add an edge

    # Debugging outputs (optional)
    print("Subgraph nodes:", len(subgraph.nodes(data=True)))
    print("Subgraph edges:", len(subgraph.edges()))
    print("Graph nodes:", len(graph.nodes(data=True)))
    print("Graph edges:", len(graph.edges()))
