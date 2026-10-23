import os
import json
import numpy as np
import copy
import torch
import random
import pandas as pd
import h5py
import pickle
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import graphein.protein as gp
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold
import io
from tqdm import tqdm
import logging
from utils import UnionFindSet, get_bfs_sub_graph, get_dfs_sub_graph
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
import requests


# hf = None
# def init_worker(hdf5_path):
#     global hf
#     hf = h5py.File(hdf5_path, 'r')

# logging.basicConfig(filename='logs/balanced_pspa_testing.log', encoding='utf-8', level=logging.INFO)

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

# def get_graph_for_uniprot(acc_id, config, folder_path="./alphafold_pdb_files"):
#     """
#     Ensures PDB file exists for UniProt ID, downloads if needed,
#     then constructs Graphein graph.
#     """
#     pdb_path = os.path.join(folder_path, f"{acc_id}.pdb")
    
#     if not os.path.isfile(pdb_path):
#         logging.info(f"PDB not found for {acc_id}. Downloading...")
#         success = download_single_pdb(acc_id, folder_path)
#         if not success:
#             logging.warning(f"❌ Failed to retrieve structure for {acc_id}")
#             return None

#     try:
#         graph = construct_graph(config=config, path=pdb_path)
#         return graph
#     except Exception as e:
#         logging.error(f"Graph construction failed for {acc_id}: {e}")
#         return None
# def extract_feature_for_kinase(name, protein_dict, config, placeholder_value=np.zeros(1280)):
#     global hf
#     if len(name.split('_')) == 1:  # Kinase
#         try:
#             pt_bytes = hf[name]["embeddings"][:]
#             buffer = io.BytesIO(pt_bytes.tobytes())
#             embeddings = torch.load(buffer, weights_only=False)
#             feature_vectors = []
#             graph = get_graph_for_uniprot(name, config)
#             for i, (_, node_data) in enumerate(graph.nodes(data=True)):
#                 if i < len(embeddings):  # Ensure embeddings match the graph nodes
#                     node_data["feature"] = embeddings[i].numpy()
#                 else:
#                     logging.info(f"Residue mismatch in {name}")
#                     return None
#             for node, feature in graph.nodes(data='feature'):
#                 if feature is not None:  # Ensure the feature exists
#                     feature_vectors.append(feature)
#             if feature_vectors:
#                 feature_array = np.array(feature_vectors)
#                 mean_feature_vector = feature_array.mean(axis=0)
#                 protein_dict[name] = mean_feature_vector
#             else:
#                 protein_dict[name] = placeholder_value
#                 logging.info(f"No features found for kinase {name}, assigning placeholder.")
#         except Exception as e:
#             protein_dict[name] = placeholder_value
#             logging.info(f"Error processing kinase {name}: {e}. Assigning placeholder.")
#     return protein_dict

# def extract_feature_for_substrate(name, protein_dict, AA_DICT, config, placeholder_value=np.zeros(1280)):
#     global hf
#     if len(name.split('_')) > 1:  # Substrate
#         try:
#             sub_id = name.split('_')[0]
#             site_sequence = name.split('_')[1]
#             site_sequence = site_sequence.upper().replace('_', '')
#             pt_bytes = hf[sub_id]["embeddings"][:]
#             buffer = io.BytesIO(pt_bytes.tobytes())
#             embeddings = torch.load(buffer, weights_only=False)
#             graph = get_graph_for_uniprot(sub_id, config)
#             for i, (_, node_data) in enumerate(graph.nodes(data=True)):
#                 if i < len(embeddings):
#                     node_data["feature"] = embeddings[i].numpy()
#                 else:
#                     logging.info(f"Residue mismatch in {name}")
#                     return None
#             total_sequence = ""
#             feature_vectors = []
#             for node, feature in graph.nodes(data='feature'):
#                 node_id = node.split(":")[1]
#                 letter = AA_DICT.get(node_id, '')
#                 total_sequence += letter
#                 feature_vectors.append(feature)
#             index = total_sequence.find(site_sequence)
#             feature_vectors = feature_vectors[index: index + len(site_sequence)]
#             if feature_vectors:
#                 feature_array = np.array(feature_vectors)
#                 mean_feature_vector = feature_array.mean(axis=0)
#                 protein_dict[name] = mean_feature_vector
#             else:
#                 protein_dict[name] = placeholder_value
#                 logging.info(f"No features found for substrate {name}, assigning placeholder.")
#         except Exception as e:
#             protein_dict[name] = placeholder_value
#             logging.info(f"Error processing substrate {name}: {e}. Assigning placeholder.")
#     return protein_dict

# def worker_task(args):
#     name, protein_dict, AA_DICT, config, placeholder_value = args
#     if '_' in name:
#         return extract_feature_for_substrate(name, protein_dict, AA_DICT, config, placeholder_value)
#     else:
#         return extract_feature_for_kinase(name, protein_dict, config, placeholder_value)

def read_mean_feature(name, hdf5_filename='./GPS/graphein_protein_data.h5'):
    try:
        with h5py.File(hdf5_filename, "r") as hf:
            if name not in hf:
                logging.warning(f"{name} not found in HDF5.")
                return None
            mean_vector = hf[name]["mean_feature"][:]
            return name, mean_vector
    except Exception as e:
        logging.error(f"Error reading {name} from HDF5: {e}")
        return None

        
class IMBALANCED_GNN_DATA:
    def __init__(self, dataset_csv, config, data_folder):
        df = pd.read_csv(dataset_csv).head(10000)
        # Initialize the lists
        self.config = config
        self.data_folder = data_folder
        self.ppi_list = []
        self.ppi_label_list = []
        self.protein_name = {}
        self.protein_dict = {}
        self.AA_DICT = {
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


        # Create a mapping of node names to indices
        node_index = 0

        # Iterate through each row in the dataframe
        # Open HDF5 file once and get all available keys
        with h5py.File('./GPS/graphein_protein_data.h5', 'r') as hf:
            available_proteins = set(hf.keys())

        # Initialize structures
        node_index = 0
        self.protein_name = {}
        self.ppi_list = []
        self.ppi_label_list = []
        self.divided_len = 0

        # Iterate through the CSV rows
        for i, row in df.iterrows():
            if i == 199999:
                self.divided_len = len(self.ppi_list)
            kinase_node = row['KIN_ACC_ID']
            substrate_node = f"{row['SUB_ACC_ID']}_{row['SITE_+/-7_AA']}"

            # Skip if either protein is missing from the HDF5
            if kinase_node not in available_proteins or substrate_node not in available_proteins:
                continue

            # Assign an index to the kinase node if it's not already in the mapping
            if kinase_node not in self.protein_name:
                self.protein_name[kinase_node] = node_index
                node_index += 1
            
            # Assign an index to the substrate node if it's not already in the mapping
            if substrate_node not in self.protein_name:
                self.protein_name[substrate_node] = node_index
                node_index += 1

            kinase_index = self.protein_name[kinase_node]
            substrate_index = self.protein_name[substrate_node]

            # Store edge and label
            self.ppi_list.append([kinase_index, substrate_index])
            self.ppi_label_list.append([row['true_label']])

        # Add reversed edges
        original_edges = len(self.ppi_list)
        for i in range(original_edges):
            edge = self.ppi_list[i]
            label = self.ppi_label_list[i]

            self.ppi_list.append(edge[::-1])
            self.ppi_label_list.append(label)

        # Final stats
        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)



    # def get_feature_origin(self):
    #     file_path = f"{self.data_folder}/protein_dict.pkl"
    #     if os.path.exists(file_path):
    #         with open(file_path, 'rb') as f:
    #             self.protein_dict = pickle.load(f)
    #         logging.info("Protein dictionary loaded from file.")
    #         return

    #     with Manager() as manager:
    #         protein_dict = manager.dict()
    #         placeholder_value = np.zeros(1280)
    #         tasks = [(name, protein_dict, self.AA_DICT, self.config, placeholder_value) for name in self.protein_name.keys()]

    #         with Pool(processes=36, initializer=init_worker, initargs=('../../../../scratch/pbsjobs/job.2634651.hpc/protein_data.h5',)) as pool:
    #             list(tqdm(pool.imap(worker_task, tasks), total=len(tasks), desc="Processing Proteins"))

    #         self.protein_dict = dict(protein_dict)
    #         with open(file_path, 'wb') as f:
    #             pickle.dump(self.protein_dict, f)
    #         logging.info("Protein dictionary saved to file.")

    def get_feature_origin(self):
        file_path = "./GPS/GPS_testing/protein_dict.pkl"
        
        # Use cached version if available
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.protein_dict = pickle.load(f)
            logging.info("Protein dictionary loaded from file.")
            return

        protein_keys = list(self.protein_name.keys())

        # Parallel HDF5 mean feature reader
        worker_func = partial(read_mean_feature)
        
        with Pool(processes=64) as pool:
            results = list(tqdm(pool.imap(worker_func, protein_keys), total=len(protein_keys), desc="Reading Mean Features"))

        # Filter and convert to dictionary
        self.protein_dict = {name: vec for name, vec in results if name and vec is not None}

        # Save to disk
        with open(file_path, 'wb') as f:
            pickle.dump(self.protein_dict, f)
        logging.info("Protein dictionary saved to file.")

    def get_connected_num(self):
        self.ufs = UnionFindSet(self.node_num)
        ppi_ndary = np.array(self.ppi_list)
        for edge in ppi_ndary:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)

    def generate_data(self):
        self.get_connected_num()

        logging.info("Connected domain num: {}".format(self.ufs.count))

        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)
        self.x = []
        i = 0
        for name in self.protein_name:
            assert self.protein_name[name] == i
            i += 1
            self.x.append(self.protein_dict[name])

        self.x = np.array(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float)

        self.data = Data(x=self.x, edge_index=self.edge_index.T, edge_attr_1=self.edge_attr)

    def split_dataset(self, train_valid_index_path, val_size = 0, test_size=1, df_divide=False, random_new=False, mode='random'):
        if df_divide:
            ppi_num = int(self.divided_len // 2)
            random_list = [i for i in range(ppi_num)]
            random.shuffle(random_list)

            self.ppi_split_dict = {}
            train_end = int(ppi_num * (.8))
            val_end = int(ppi_num * (.2))
            self.ppi_split_dict['train_index'] = random_list[:train_end]
            self.ppi_split_dict['valid_index'] = random_list[train_end:val_end]
            test_start = self.divided_len
            test_end = len(self.ppi_list)
            self.ppi_split_dict['test_index'] = list(range(test_start, test_end))

            jsobj = json.dumps(self.ppi_split_dict)
            with open(train_valid_index_path, 'w') as f:
                f.write(jsobj)
                f.close()
        if random_new:
            if mode == 'random':
                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                train_end = int(ppi_num * (1 - test_size - val_size))
                val_end = int(ppi_num * (1 - test_size))
                self.ppi_split_dict['train_index'] = random_list[:train_end]
                self.ppi_split_dict['valid_index'] = random_list[train_end:val_end]
                self.ppi_split_dict['test_index'] = random_list[val_end:]

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            elif mode == 'bfs' or mode == 'dfs':
                logging.info("use {} methed split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = unselected_edge_index
                self.ppi_split_dict['valid_index'] = selected_edge_index

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            else:
                logging.info("your mode is {}, you should use bfs, dfs or random".format(mode))
                return
        else:
            with open(train_valid_index_path, encoding='utf-8-sig',errors='ignore') as f:
                str = f.read()
                self.ppi_split_dict = json.loads(str, strict=False)
                f.close()