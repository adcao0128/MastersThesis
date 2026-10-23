import pandas as pd
import re
import logging
import graphein.protein as gp
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold, add_peptide_bonds
import networkx as nx
import itertools
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import io
import sys
from functools import partial

#This cannot be run on the dodeca96gb cpu node

logging.basicConfig(filename='./logs/sequence_1hop_stats.log', encoding='utf-8', level=logging.INFO)

graph_type = "khop"

new_edge_funcs = {"edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=5, threshold=10.), add_peptide_bonds]}
config = ProteinGraphConfig(**new_edge_funcs)

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

def read_psp_dataset(filepath="Kinase_Substrate_Dataset.txt"):
    df = pd.read_csv(filepath, sep='\t')
    df = df[["KIN_ACC_ID", "KIN_ORGANISM", "SUB_ACC_ID", "SITE_+/-7_AA"]]
    df = df[df["KIN_ORGANISM"] == "human"]

    return df

def clean_txt_from_pdb(df, log_file="./logs/alphafold_query.log", another_log_file="./logs/archived_structure_backbone.log"):
    data = []

    pattern = r"url: https://alphafold.ebi.ac.uk/api/prediction/(\S+)"

    # Read the file line by line
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                # Extract the data from the matched groups and append to the list
                graph_id, = match.groups()
                data.append(graph_id)

    another_pattern = r"INFO:root:Residue mismatch in (\S+)"
    another_third = r"INFO:root:Residue mismatch in ([^_]+)_+(\S+)"
    fourth = r"INFO:root:No features found for substrate ([^_]+)_+(\S+), assigning placeholder."
    second_data = []
    # Read the file line by line
    with open(another_log_file, 'r') as f:
        for line in f:
            second_match = re.search(another_pattern, line)
            if second_match:
                # Extract the data from the matched groups and append to the list
                graph_id, = second_match.groups()
                second_data.append(graph_id)
            third_match = re.search(another_third, line)
            if third_match:
                # Extract the data from the matched groups and append to the list
                graph_id, sequence = third_match.groups()
                second_data.append(graph_id)
            fourth_match = re.search(fourth, line)
            if fourth_match:
                # Extract the data from the matched groups and append to the list
                graph_id, sequence = fourth_match.groups()
                second_data.append(graph_id)
    # Create a pandas DataFrame from the extracted data
    data.extend(second_data)
    values_to_remove = pd.DataFrame(data)
    # Populate protein_name with indices
    for index, row in values_to_remove.iterrows():
        value = row[0]
        df = df[df.KIN_ACC_ID != value]
        df = df[df.SUB_ACC_ID != value]
    return df


def process_site_sequence(args):
    acc_id, site_sequence = args
    graph = construct_graph(config=config, path=str(f"./alphafold_pdb_files/{acc_id}.pdb"))


    if graph_type == "khop":
        k = 1  # Define the number of hops

        # Step 1: Extract main sequence and features
        total_sequence = ""
        node_list = []  # Maintain order of original nodes

        for node in graph.nodes():
            node_id = node.split(":")[1]  # Extract amino acid ID
            letter = AA_DICT[node_id]  # Convert to amino acid
            total_sequence += letter
            node_list.append(node)  # Maintain order

        # Step 2: Find start index of the subsequence
        index = total_sequence.find(site_sequence)
        if index == -1:
            logging.info(f"Site sequence '{site_sequence}' not found in protein sequence {acc_id}.")
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
        logging.info(f"Site Sequence: {site_sequence}, Structure Edges: {len(subgraph.edges())}, Total Nodes: {len(subgraph.nodes())}")

    if graph_type == "structure":
        # Extract sequence and features
        total_sequence = ""
        node_list = []  # Maintain order of original nodes

        for node in graph.nodes():
            node_id = node.split(":")[1]  # Extract amino acid ID
            letter = AA_DICT[node_id]  # Convert to amino acid
            total_sequence += letter
            node_list.append(node)  # Maintain order

        # Find start index of the subsequence
        index = total_sequence.find(site_sequence)
        if index == -1:
            logging.info(f"Site sequence '{site_sequence}' not found in protein sequence {acc_id}.")

        # Extract the relevant nodes and features
        selected_nodes = node_list[index: index + len(site_sequence)]

        # Create a mapping from original node IDs to new indices
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}

        # Extract and remap edges
        selected_edges = [
            (node_mapping[u], node_mapping[v]) 
            for u, v in graph.edges() 
            if u in selected_nodes and v in selected_nodes
        ]

        # Create the subgraph with new indices
        subgraph = nx.Graph()  # Use nx.DiGraph() if directed
        for new_id, (old_id) in enumerate(zip(selected_nodes)):
            subgraph.add_node(new_id, original_id=old_id)  # Store old ID if needed

        subgraph.add_edges_from(selected_edges)  # Add remapped edges

        # Optional: Check the subgraph
        # print("Subgraph nodes:", subgraph.nodes(data=True))
        # print("Subgraph edges:", subgraph.edges())

        logging.info(f"Graph ID: {site_sequence}, Structure Edges: {len(subgraph.edges())}, Total Nodes: {len(subgraph.nodes())}")

    if graph_type == "2hop":
        k = 2  # Define the number of hops

        # Step 1: Extract main sequence and features
        total_sequence = ""
        node_list = []  # Maintain order of original nodes

        for node in graph.nodes():
            node_id = node.split(":")[1]  # Extract amino acid ID
            letter = AA_DICT[node_id]  # Convert to amino acid
            total_sequence += letter
            node_list.append(node)  # Maintain order

        # Step 2: Find start index of the subsequence
        index = total_sequence.find(site_sequence)
        if index == -1:
            logging.info(f"Site sequence '{site_sequence}' not found in protein sequence {acc_id}.")
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
        logging.info(f"Site Sequence: {site_sequence}, Structure Edges: {len(subgraph.edges())}, Total Nodes: {len(subgraph.nodes())}")



def process_all_site_sequences(graph_type, df, pdb_folder="./alphafold_pdb_files/"):
    
    tasks = []
    for index, row in df.iterrows():
        acc_id = row["SUB_ACC_ID"]
        site_sequence = row["SITE_+/-7_AA"].upper().replace('_', '')
        tasks.append((acc_id, site_sequence))

    with Pool(processes=cpu_count()) as pool:  # Use 30 workers
        results = list(tqdm(pool.imap(process_site_sequence,tasks ), total=len(tasks), desc="Processing Site Sequences"))

        

        


if __name__ == "__main__":
    text_trap = io.StringIO()
    sys.stdout = text_trap
    df = read_psp_dataset()
    print(df.info())
    df = clean_txt_from_pdb(df)
    print(df.info())

    process_all_site_sequences(graph_type, df)