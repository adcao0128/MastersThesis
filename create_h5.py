import os
import pandas as pd
import re
import logging
import networkx as nx
import itertools
from multiprocessing import Pool, Process, cpu_count, Manager
from tqdm import tqdm
import io
import sys
import os
import h5py
import torch
import gzip
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import pickle
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold, add_peptide_bonds
from functools import partial
from more_itertools import chunked
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import numpy as np

new_edge_funcs = {"edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=5, threshold=10.), add_peptide_bonds]}
config = ProteinGraphConfig(**new_edge_funcs)

logging.basicConfig(filename='create_h5.log', encoding='utf-8', level=logging.INFO)
pdb_folder = '../job.2751278.pioneer/alphafold_pdb_files'
embeddings_folder = '../job.2751278.pioneer/residue_embeddings'

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

def save_graphs_to_hdf5(uq_acc_ids, hdf5_filename):
    with h5py.File(hdf5_filename, 'w') as hf:
        for acc_id in uq_acc_ids:
            print(f"processing {acc_id}")
            protein_group = hf.create_group(f"{acc_id}")
            with open(f"./alphafold_pdb_files/{acc_id}.pdb", "rb") as f:
                compressed_pdb = gzip.compress(f.read())
                protein_group.create_dataset("pdb", data=np.frombuffer(compressed_pdb, dtype=np.uint8))

            pt_tensor = torch.load(f"./residue_embeddings/{acc_id}.pt")

            buffer = io.BytesIO()
            torch.save(pt_tensor, buffer)
            buffer.seek(0)
            
            pt_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

            protein_group.create_dataset(
                "pt", 
                data=pt_bytes, 
                compression="gzip", 
                compression_opts=9
            )

def load_embeddings(embedding_path):
    """Load residue-level embeddings from ESM-2 output."""
    data = torch.load(embedding_path, weights_only=False)
    return data["representations"][33]

def process_and_store(acc_id, hf):
    logging.info(f"Processing {acc_id}")
    protein_group = hf.create_group(f"{acc_id}")

    # Process and store PDB (compressed)
    pdb_path = os.path.join(pdb_folder, f"{acc_id}.pdb")
    with open(pdb_path, "rb") as f:
        compressed_pdb = gzip.compress(f.read())
        protein_group.create_dataset("pdb", data=np.frombuffer(compressed_pdb, dtype=np.uint8))

    # Process and store PT (compressed)
    pt_path = os.path.join(embeddings_folder, f"{acc_id}.pt")
    embedding = load_embeddings(pt_path)

    buffer = io.BytesIO()
    torch.save(embedding, buffer)
    buffer.seek(0)

    pt_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    protein_group.create_dataset("embeddings", data=pt_bytes, compression="gzip", compression_opts=9)

def save_graphs_to_hdf5(uq_acc_ids, hdf5_filename, num_threads=24):
    with h5py.File(hdf5_filename, "w", libver="latest") as hf:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Pass `hf` as an argument, since it must be managed within a single process
            executor.map(lambda acc_id: process_and_store(acc_id, hf), uq_acc_ids)

# def read_psp_dataset(filepath="Kinase_Substrate_Dataset.txt"):
#     df = pd.read_csv(filepath, sep='\t')
#     df = df[["KIN_ACC_ID", "KIN_ORGANISM", "SUB_ACC_ID", "SITE_+/-7_AA"]]
#     df = df[df["KIN_ORGANISM"] == "human"]

#     return df

# def clean_txt_from_pdb(df, log_file="./logs/alphafold_query.log"):
#     data = []

#     pattern = r"url: https://alphafold.ebi.ac.uk/api/prediction/(\S+)"

#     # Read the file line by line
#     with open(log_file, 'r') as f:
#         for line in f:
#             match = re.search(pattern, line)
#             if match:
#                 # Extract the data from the matched groups and append to the list
#                 graph_id = match.groups()
#                 data.append({
#                     'GraphID': graph_id
#                 })

#     # Create a pandas DataFrame from the extracted data
#     values_to_remove = pd.DataFrame(data)


#     # Populate protein_name with indices
#     for index, row in values_to_remove.iterrows():
#         value = row["GraphID"]
#         value = value[0]
#         df = df[df["KIN_ACC_ID"] != value]
#         df = df[df["SUB_ACC_ID"] != value]
    
#     return df

# def get_uq_acc_ids(df):
#     uq_acc_ids = pd.unique(df[['KIN_ACC_ID', 'SUB_ACC_ID']].values.ravel())
#     return uq_acc_ids

# def get_protein(uniprot_accession):
#     """Query AlphaFold API for a UniProt ID."""
#     api_endpoint = "https://alphafold.ebi.ac.uk/api/prediction/"
#     url = f"{api_endpoint}{uniprot_accession}"

#     try:
#         response = requests.get(url, timeout=10)
#         if response.status_code == 200:
#             return response.json()
#         else:
#             response.raise_for_status()
#     except requests.exceptions.RequestException as e:
#         logging.info(f"Error for {uniprot_accession}: {e}")
#         return None

# def download_single_pdb(acc_id, folder_path="./alphafold_pdb_files"):
#     """Download AlphaFold PDB from EBI and save it."""
#     os.makedirs(folder_path, exist_ok=True)
#     protein_info = get_protein(acc_id)
#     if protein_info:
#         pdb_url = protein_info[0].get('pdbUrl')
#         if pdb_url:
#             try:
#                 pdb_data = requests.get(pdb_url, timeout=10).content
#                 with open(f"{folder_path}/{acc_id}.pdb", mode="wb") as file:
#                     file.write(pdb_data)
#                 logging.info(f"✅ Downloaded PDB for {acc_id}")
#                 return True
#             except Exception as e:
#                 logging.info(f"Download failed for {acc_id}: {e}")
#         else:
#             logging.info(f"No PDB URL for {acc_id}.")
#     else:
#         logging.info(f"No protein info for {acc_id}.")
#     return False

def get_graph_for_uniprot(acc_id, config, folder_path=pdb_folder):
    """
    Ensures PDB file exists for UniProt ID, downloads if needed,
    then constructs Graphein graph.
    """
    pdb_path = os.path.join(folder_path, f"{acc_id}.pdb")
    
    if not os.path.isfile(pdb_path):
        logging.info(f"PDB not found for {acc_id}. Downloading...")
        success = download_single_pdb(acc_id, folder_path)
        if not success:
            logging.warning(f"❌ Failed to retrieve structure for {acc_id}")
            return None

    try:
        graph = construct_graph(config=config, path=pdb_path)
        return graph
    except Exception as e:
        logging.error(f"Graph construction failed for {acc_id}: {e}")
        return None

def extract_feature_for_kinase(name, placeholder_value, hdf5_filename):
    try:
        with h5py.File(hdf5_filename, "r") as hf:
            pt_bytes = hf[name]["embeddings"][:]
        buffer = io.BytesIO(pt_bytes.tobytes())
        embeddings = torch.load(buffer, weights_only=False)

        graph = get_graph_for_uniprot(name, config)
        if graph is None:
            raise ValueError("Graph is None")

        feature_vectors = []
        for i, (_, node_data) in enumerate(graph.nodes(data=True)):
            if i < len(embeddings):
                feature_vectors.append(embeddings[i].numpy())
            else:
                logging.info(f"Residue mismatch in {name}")
                return None

        if not feature_vectors:
            logging.info(f"No features for {name}")
            return None

        feature_array = np.array(feature_vectors)
        mean_vector = feature_array.mean(axis=0)
        node_index_map = {node: i for i, node in enumerate(graph.nodes())}
        indexed_edges = [(node_index_map[u], node_index_map[v]) for u, v in graph.edges()]

        return name, mean_vector, feature_array, indexed_edges

    except Exception as e:
        logging.info(f"Error processing kinase {name}: {e}")
        return None


def extract_feature_for_substrate(name, placeholder_value, hdf5_filename):
    try:
        sub_id, site_sequence = name.split('_', 1)
        site_sequence = site_sequence.upper().replace('_', '')

        with h5py.File(hdf5_filename, "r") as hf:
            pt_bytes = hf[sub_id]["embeddings"][:]

        buffer = io.BytesIO(pt_bytes.tobytes())
        embeddings = torch.load(buffer, weights_only=False)

        graph = get_graph_for_uniprot(sub_id, config)
        if graph is None:
            raise ValueError("Graph is None")

        total_sequence = ""
        feature_vectors = []
        node_list = []

        for i, (node, node_data) in enumerate(graph.nodes(data=True)):
            if i < len(embeddings):
                feature = embeddings[i].numpy()
                feature_vectors.append(feature)
                node_list.append(node)
                node_data["feature"] = feature
            else:
                logging.info(f"Residue mismatch in {name}")
                return None

            node_id = node.split(":")[1]
            total_sequence += AA_DICT.get(node_id, '')

        index = total_sequence.find(site_sequence)
        if index == -1:
            logging.info(f"Site sequence '{site_sequence}' not found in {sub_id}")
            return None

        selected_nodes = node_list[index: index + len(site_sequence)]
        selected_features = feature_vectors[index: index + len(site_sequence)]

        # Build subgraph based on shared 1-hop neighborhoods
        k = 1
        node_to_k_hop = {
            node: set(nx.single_source_shortest_path_length(graph, node, cutoff=k).keys())
            for node in selected_nodes
        }

        subgraph = nx.Graph()
        subgraph.add_nodes_from(selected_nodes)
        for n1, n2 in itertools.combinations(selected_nodes, 2):
            if node_to_k_hop[n1] & node_to_k_hop[n2]:
                subgraph.add_edge(n1, n2)

        # Remap nodes to 0-based indices
        node_mapping = {old: new for new, old in enumerate(selected_nodes)}
        indexed_edges = [(node_mapping[u], node_mapping[v]) for u, v in subgraph.edges()]

        feature_array = np.array(selected_features)
        mean_vector = feature_array.mean(axis=0)

        return name, mean_vector, feature_array, indexed_edges

    except Exception as e:
        logging.info(f"Error processing substrate {name}: {e}")
        return None


def worker_task(name, hdf5_read_filename, placeholder_value):
    # Extract features
    if '_' in name:
        result = extract_feature_for_substrate(name, placeholder_value, hdf5_read_filename)
    else:
        result = extract_feature_for_kinase(name, placeholder_value, hdf5_read_filename)

    if result is None:
        logging.info(f"Failed to extract features for {name}")
        return None

    name, mean_vector, feature_array, indexed_edges = result

    return (name, mean_vector, feature_array, indexed_edges)



def write_results_to_hdf5(results, hdf5_write_filename):
    with h5py.File(hdf5_write_filename, "a", libver="latest") as hf_write:
        for item in results:
            if item is None:
                continue

            name, mean_vector, feature_array, indexed_edges = item

            if name in hf_write:
                logging.info(f"Group {name} already exists in HDF5, skipping.")
                continue

            group = hf_write.create_group(name)
            group.create_dataset("mean_feature", data=mean_vector, compression="gzip")
            group.create_dataset("feature_array", data=feature_array, compression="gzip")
            group.create_dataset("indexed_edges", data=np.array(indexed_edges, dtype='int32'))

            print(f"Saved {name}")



def save_graphein_to_hdf5(hdf5_read_filename, hdf5_write_filename, num_processes=40, chunk_size=1000):
    df = pd.read_csv('GPS_psp_7_7_alphafold_checked.csv')
    print("Reading CSV and collecting proteins...")

    protein_name = {}
    node_index = 0

    for _, row in df.iterrows():
        kinase_node = row['KIN_ACC_ID']
        substrate_node = f"{row['SUB_ACC_ID']}_{row['SITE_+/-7_AA']}"
        if kinase_node not in protein_name:
            protein_name[kinase_node] = node_index
            node_index += 1
        if substrate_node not in protein_name:
            protein_name[substrate_node] = node_index
            node_index += 1

    print(f"Total unique proteins: {len(protein_name)}")

    placeholder_value = np.zeros(1280)
    worker_func = partial(
        worker_task,
        hdf5_read_filename=hdf5_read_filename,
        placeholder_value=placeholder_value
    )

    protein_keys = list(set(protein_name.keys()))

    for chunk in chunked(protein_keys, chunk_size):
        print("Finished a chunk")
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(worker_func, chunk), total=len(chunk), desc="Processing Proteins"))
        write_results_to_hdf5(results, hdf5_write_filename)

print("test STARTING create_h5.py", flush=True)
if __name__ == "__main__":
    # text_trap = io.StringIO()
    # sys.stdout = text_trap
    # df = read_psp_dataset()
    # logging.info(df.info())
    # df = clean_txt_from_pdb(df)
    # logging.info(df.info())
    # uq_acc_ids = get_uq_acc_ids(df)
    # logging.info(len(uq_acc_ids))
    hdf5_filename = '../job.2755968.pioneer/protein_data.h5'
    graphein_hdf5_filename = 'graphein_protein_data.h5'

    # pdb_ids = {f.split('.')[0] for f in os.listdir(pdb_folder) if f.endswith('.pdb')}
    # pt_ids = {f.split('.')[0] for f in os.listdir(embeddings_folder) if f.endswith('.pt')}
    # uq_acc_ids = sorted(pdb_ids & pt_ids)
    # print(len(uq_acc_ids))
    # print(len(uq_acc_ids), flush=True)

    # save_graphs_to_hdf5(uq_acc_ids, hdf5_filename)

    save_graphein_to_hdf5(hdf5_filename, graphein_hdf5_filename)