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

logging.basicConfig(filename='logs/final_test.log', encoding='utf-8', level=logging.INFO)

class GNN_DATA:
    def __init__(self, dataset_csv, config, data_folder):
        df = pd.read_csv(dataset_csv)
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
        for index, row in df.iterrows():
            # Create nodes
            kinase_node = row['KIN_ACC_ID']
            substrate_node = f"{row['SUB_ACC_ID']}_{row['SITE_+/-7_AA']}"
            
            # Assign an index to the kinase node if it's not already in the mapping
            if kinase_node not in self.protein_name:
                self.protein_name[kinase_node] = node_index
                node_index += 1
            
            # Assign an index to the substrate node if it's not already in the mapping
            if substrate_node not in self.protein_name:
                self.protein_name[substrate_node] = node_index
                node_index += 1
            
            # Get the node indices
            kinase_node_index = self.protein_name[kinase_node]
            substrate_node_index = self.protein_name[substrate_node]
            
            # Create the edge as a list of node indices
            edge = [kinase_node_index, substrate_node_index]
            
            # Append the edge to ppi_list
            self.ppi_list.append(edge)
            
            # Append the true label to ppi_label_list
            self.ppi_label_list.append([row['true_label']])

        for i in range(len(self.ppi_list)):
            edge = self.ppi_list[i]
            label = self.ppi_label_list[i]
            
            # Add the reversed edge
            reversed_edge = edge[::-1]
            self.ppi_list.append(reversed_edge)
            self.ppi_label_list.append(label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

    def extract_feature_for_kinase(self, name, protein_dict, placeholder_value=np.zeros(1280)):  # Specify the size of the placeholder if needed
        if len(name.split('_')) == 1:  # Kinase
            with h5py.File('../protein_data.h5', 'r') as hf:
                pt_bytes = hf[name]["embeddings"][:]
                buffer = io.BytesIO(pt_bytes.tobytes())
                embeddings = torch.load(buffer, weights_only=False)
                try:
                    feature_vectors = []
                    graph = construct_graph(config=self.config, path=f"../alphafold_pdb_files/{name}.pdb")
                    for i, (_, node_data) in enumerate(graph.nodes(data=True)):
                        if i < len(embeddings):  # Ensure embeddings match the graph nodes
                            node_data["feature"] = embeddings[i].numpy()
                        else:
                            logging.info(f"Residue mismatch in {name}")
                            return None
                    for node, feature in graph.nodes(data='feature'):
                        if feature is not None:  # Ensure the feature exists
                            feature_vectors.append(feature)
                    if feature_vectors:  # If there are features
                        feature_array = np.array(feature_vectors)
                        mean_feature_vector = feature_array.mean(axis=0)
                        protein_dict[name] = mean_feature_vector
                    else:  # If no features found, assign the placeholder
                        protein_dict[name] = placeholder_value
                        logging.info(f"No features found for kinase {name}, assigning placeholder.")
                except Exception as e:
                    protein_dict[name] = placeholder_value
                    logging.info(f"Error processing kinase {name}: {e}. Assigning placeholder.")
        return protein_dict

    def extract_feature_for_substrate(self, name, protein_dict, AA_DICT, placeholder_value=np.zeros(1280)):  # Specify the size of the placeholder if needed
        if len(name.split('_')) > 1:  # Substrate
            sub_id = name.split('_')[0]
            site_sequence = name.split('_')[1]
            site_sequence = site_sequence.upper().replace('_', '')
            with h5py.File('../protein_data.h5', 'r') as hf:
                pt_bytes = hf[sub_id]["embeddings"][:]
                buffer = io.BytesIO(pt_bytes.tobytes())
                embeddings = torch.load(buffer, weights_only=False)
                try:
                    graph = construct_graph(config=self.config, path=f"../alphafold_pdb_files/{sub_id}.pdb")
                    for i, (_, node_data) in enumerate(graph.nodes(data=True)):
                        if i < len(embeddings):  # Ensure embeddings match the graph nodes
                            node_data["feature"] = embeddings[i].numpy()
                        else:
                            logging.info(f"Residue mismatch in {name}")
                            return None
                    total_sequence = ""
                    feature_vectors = []
                    for node, feature in graph.nodes(data='feature'):
                        node_id = node.split(":")[1]
                        letter = AA_DICT[node_id]
                        total_sequence += letter
                        feature_vectors.append(feature)
                    index = total_sequence.find(site_sequence)
                    feature_vectors = feature_vectors[index: index + len(site_sequence)]
                    if feature_vectors:  # If there are features
                        feature_array = np.array(feature_vectors)
                        mean_feature_vector = feature_array.mean(axis=0)
                        protein_dict[name] = mean_feature_vector
                    else:  # If no features found, assign the placeholder
                        protein_dict[name] = placeholder_value
                        logging.info(f"No features found for substrate {name}, assigning placeholder.")
                except Exception as e:
                    protein_dict[name] = placeholder_value
                    logging.info(f"Error processing substrate {name}: {e}. Assigning placeholder.")
        return protein_dict

    def get_feature_for_protein(self, args):
        name, protein_dict, AA_DICT = args  # Unpack the tuple
        if len(name.split('_')) == 1:
            return self.extract_feature_for_kinase(name, protein_dict)
        else:
            return self.extract_feature_for_substrate(name, protein_dict, AA_DICT)

    def get_feature_origin(self):
        file_path = f"{self.data_folder}/protein_dict.pkl"  # Path to save/load the dictionary

        # Check if the file exists
        if os.path.exists(file_path):
            # Load the dictionary from the file
            with open(file_path, 'rb') as f:
                self.protein_dict = pickle.load(f)
            logging.info("Protein dictionary loaded from file.")
        else:
            # Use manager to create a shared dictionary for multiprocessing
            with Manager() as manager:
                protein_dict = manager.dict()
                tasks = [(name, protein_dict, self.AA_DICT) for name in self.protein_name.keys()]

                # Use partial to pass 'self' to 'get_feature_for_protein'
                with Pool(processes=cpu_count()) as pool:
                    func_with_self = partial(self.get_feature_for_protein)

                    # Use tqdm to wrap pool.imap for progress tracking
                    list(tqdm(pool.imap(func_with_self, tasks), total=len(tasks), desc="Processing Proteins"))

                # Directly update self.protein_dict from the manager's shared protein_dict
                self.protein_dict = dict(protein_dict)

                # Save the dictionary to a file for future use
                with open(file_path, 'wb') as f:
                    pickle.dump(self.protein_dict, f)
                logging.info("Protein dictionary saved to file.")

        logging.info("Finished updating protein dictionary.")

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

    def split_dataset(self, train_valid_index_path, val_size = .1, test_size=0.2, random_new=False, mode='random'):
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