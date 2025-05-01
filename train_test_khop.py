import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
from GNN_DATA import GNN_DATA
from model import ppi_model
from utils import Metrictor_PPI, print_file
from tensorboardX import SummaryWriter
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import pickle
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold, add_peptide_bonds
from functools import partial
import io
import sys
import logging
import pandas as pd
import h5py
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

logging.basicConfig(filename='logs/final_test.log', encoding='utf-8', level=logging.INFO)

new_edge_funcs = {"edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=5, threshold=10.), add_peptide_bonds]}
config = ProteinGraphConfig(**new_edge_funcs)

seed_num = 2
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
data_folder="./final_test"

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

def get_features_and_edgelist_all():
    df = pd.read_csv('balanced_dataset.csv')
    protein_name = {}
    node_index = 0

    for index, row in df.iterrows():
        kinase_node = row['KIN_ACC_ID']
        substrate_node = f"{row['SUB_ACC_ID']}_{row['SITE_+/-7_AA']}"

        if kinase_node not in protein_name:
            protein_name[kinase_node] = node_index
            node_index += 1

        if substrate_node not in protein_name:
            protein_name[substrate_node] = node_index
            node_index += 1

    feature_list_all = []
    edge_list_all = []

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_protein, protein_name.keys()), total=len(protein_name)))

    logging.info(len(results))

    for res in results:
        if res:
            feature_list_all.append(res[0])
            edge_list_all.append(res[1])

    logging.info(len(feature_list_all))

    return feature_list_all, edge_list_all

def process_protein(name):
    with h5py.File('../protein_data.h5', 'r') as hf:
        if len(name.split('_')) == 1:
            pt_bytes = hf[name]["embeddings"][:]
            buffer = io.BytesIO(pt_bytes.tobytes())
            embeddings = torch.load(buffer, weights_only=False)
            protein = construct_graph(config=config, path=f"../alphafold_pdb_files/{name}.pdb")
            for i, (_, node_data) in enumerate(protein.nodes(data=True)):
                if i < len(embeddings):  # Ensure embeddings match the graph nodes
                    node_data["feature"] = embeddings[i].numpy()
                else:
                    logging.info(f"Residue mismatch in {name}")
                    return None
        else:
            sub_id = name.split('_')[0]
            site_sequence = name.split('_')[1]
            site_sequence = site_sequence.upper().replace('_', '')
            pt_bytes = hf[sub_id]["embeddings"][:]
            buffer = io.BytesIO(pt_bytes.tobytes())
            embeddings = torch.load(buffer, weights_only=False)
            protein = construct_graph(config=config, path=f"../alphafold_pdb_files/{sub_id}.pdb")
            for i, (_, node_data) in enumerate(protein.nodes(data=True)):
                if i < len(embeddings):  # Ensure embeddings match the graph nodes
                    node_data["feature"] = embeddings[i].numpy()
                else:
                    logging.info(f"Residue mismatch in {name}")
                    return None
            k = 1
            total_sequence = ""
            feature_vectors = []
            node_list = []  # Maintain order of original nodes

            for node, feature in protein.nodes(data='feature'):
                node_id = node.split(":")[1]  # Extract amino acid ID
                letter = AA_DICT[node_id]  # Convert to amino acid
                total_sequence += letter
                feature_vectors.append(feature)
                node_list.append(node)  # Maintain order

            # Find start index of the subsequence
            index = total_sequence.find(site_sequence)
            if index == -1:
                logging.info(f"Site sequence '{site_sequence}' not found in protein sequence {sub_id}.")
                features = np.zeros((1, 1280))  # Placeholder 1-node zero vector
                indexed_edges = []  # No edges
                return features, indexed_edges

            # Extract the relevant nodes and features
            selected_nodes = node_list[index: index + len(site_sequence)]
            selected_features = feature_vectors[index: index + len(site_sequence)]

            node_to_k_hop_neighbors = {
                node: set(nx.single_source_shortest_path_length(protein, node, cutoff=k).keys())
                for node in selected_nodes
            }

            subgraph = nx.Graph()  # Use nx.DiGraph() if the original graph is directed
            subgraph.add_nodes_from(selected_nodes)

            for node1, node2 in itertools.combinations(selected_nodes, 2):
                shared_neighbors = node_to_k_hop_neighbors[node1] & node_to_k_hop_neighbors[node2]
                if shared_neighbors:  # If they share at least one k-hop neighbor
                    subgraph.add_edge(node1, node2)

            # Create a mapping from original node IDs to new indices
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}

            # Extract and remap edges
            selected_edges = [
                (node_mapping[u], node_mapping[v]) 
                for u, v in subgraph.edges() 
                if u in selected_nodes and v in selected_nodes
            ]

            # Create the subgraph with new indices
            new_subgraph = nx.Graph()  # Use nx.DiGraph() if directed
            for new_id, (old_id, feature) in enumerate(zip(selected_nodes, selected_features)):
                new_subgraph.add_node(new_id, feature=feature, original_id=old_id)  # Store old ID if needed

            new_subgraph.add_edges_from(selected_edges)  # Add remapped edges

            # Optional: Check the subgraph
            # print("Subgraph nodes:", subgraph.nodes(data=True))
            # print("Subgraph edges:", subgraph.edges())
            protein = new_subgraph

    # Process the protein and extract features and edges
    features = np.array([protein.nodes[node]["feature"] for node in protein.nodes()])
    node_index_map = {node: i for i, node in enumerate(protein.nodes())}
    indexed_edges = [(node_index_map[edge[0]], node_index_map[edge[1]]) for edge in protein.edges]

    return features, indexed_edges



def multi2big_x(x_ori):
    # Calculate total number of feature vectors across all inner lists
    total_vectors = sum(len(inner_list) for inner_list in x_ori)
    logging.info(len(x_ori))
    
    # Allocate tensors for concatenation and indexing
    x_cat = torch.zeros((total_vectors, 1280))
    x_num_index = torch.zeros(len(x_ori), dtype=torch.int32)  # Tracks size of each inner list
    
    idx = 0
    for i, inner_list in enumerate(x_ori):
        if i % 500 == 0:
            logging.info(i)
        num_vectors = len(inner_list)
        x_num_index[i] = num_vectors
        
        x_cat[idx:idx + num_vectors] = torch.tensor(inner_list)
        idx += num_vectors
    

    return x_cat, x_num_index, len(x_ori)

def multi2big_batch(x_num_index, num_proteins):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,num_proteins):#14742
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch

def multi2big_edge(edge_ori, num_index):
    # Precompute the total number of edges across all lists
    total_edges = sum(len(edges) for edges in edge_ori)
    
    # Preallocate tensors
    edge_cat = torch.zeros((2, total_edges), dtype=torch.long)  # Total edges x 2 for edge pairs
    edge_num_index = torch.zeros(len(edge_ori), dtype=torch.int32)  # Size for each list
    idx = 0
    for i, edges in enumerate(edge_ori):
        if edges:
            # Proceed to convert edges to tensor
            edges_tensor = torch.tensor(np.asarray(edges).T, dtype=torch.long)
        else:
            # If no edges, still create an empty tensor but with correct shape (2, 0)
            edges_tensor = torch.empty((2, 0), dtype=torch.long)

        edge_num_index[i] = edges_tensor.size(1)
        
        # Calculate the offset for current edges
        offset = torch.sum(torch.tensor(num_index[:i])) if i > 0 else 0
        

        edge_cat[:, idx:idx + edges_tensor.size(1)] = edges_tensor + offset
        idx += edges_tensor.size(1)
    
    return edge_cat, edge_num_index


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def train(batch, p_x_all, p_edge_all, model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=512, epochs=100, scheduler=None,
          got=False):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0
    truth_edge_num = graph.edge_index.shape[1] // 2
    count = 1

    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)

        for step in range(steps):
            if step == steps - 1:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size:]
                else:
                    train_edge_id = graph.train_mask[step * batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size: step * batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            if got:
                output = model(batch, p_x_all, p_edge_all, graph.edge_index_got, train_edge_id)
                label = graph.edge_attr_got[train_edge_id]
            else:
                output = model(batch, p_x_all, p_edge_all, graph.edge_index, train_edge_id)
                label = graph.edge_attr_1[train_edge_id]

            label = label.type(torch.FloatTensor).to(device)

            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data, m(output).cpu().data)

            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1
            print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                       .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))

        valid_pre_result_list = []
        valid_label_list = []
        true_prob_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

                output = model(batch, p_x_all, p_edge_all, graph.edge_index, valid_edge_id)
                label = graph.edge_attr_1[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                loss = loss_fn(output, label)
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
                true_prob_list.append(m(output).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim = 0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.get_last_lr()),
                       save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, 'gnn_model_valid_best.ckpt'))

        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)

        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch, AUC: {}, AUPR: {}"
                .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1,
                        global_best_valid_f1, global_best_valid_f1_epoch, metrics.Auc, metrics.Aupr), save_file_path=result_file_path)


def load_file(file_name):
    # Load the tensor from the .pt file
    return torch.load(file_name, weights_only=False)

def test(ppi_data, model, graph, test_mask, device, batch, p_x_all, p_edge_all, top_k=20):
    valid_pre_result_list = []
    valid_label_list = []
    
    model.eval()
    
    batch_size = 64
    valid_steps = math.ceil(len(test_mask) / batch_size)
    
    valid_pre_result_list = []
    valid_label_list = []
    true_prob_list = []
    
    # Create a list of all kinases
    kinase_nodes = [node for node, idx in ppi_data.protein_name.items() if '_' not in node]

    for step in tqdm(range(valid_steps)):
        if step == valid_steps-1:
            valid_edge_id = test_mask[step*batch_size:]
        else:
            valid_edge_id = test_mask[step*batch_size : step*batch_size + batch_size]

        # Get output probabilities for each test edge
        output = model(batch, p_x_all, p_edge_all, graph.edge_index, valid_edge_id)
        label = graph.edge_attr_1[valid_edge_id]
        label = label.type(torch.FloatTensor).to(device)
        
        m = nn.Sigmoid()
        prob = m(output).cpu().data
        
        valid_pre_result_list.append(prob)
        valid_label_list.append(label.cpu().data)
        true_prob_list.append(prob)

        # Rank all kinase-target pairs for the current batch of substrates
        ranked_results = []
        for target_id in valid_edge_id:
            # For each target, get the predictions for all kinase-target pairs
            target_probs = []
            for kinase_node_idx in kinase_nodes:
                # Create an edge with each kinase node and the current target
                edge_idx = torch.tensor([[kinase_node_idx, target_id]])
                edge_output = model(batch, p_x_all, p_edge_all, graph.edge_index, edge_idx)
                target_probs.append(m(edge_output).cpu().data.item())

            # Rank kinase-target pairs by probability
            sorted_indices = sorted(range(len(target_probs)), key=lambda x: target_probs[x], reverse=True)
            
            # Check if the correct kinase is in the top-k
            correct_kinase_idx = [kinase_nodes.index(edge[0]) for edge in self.ppi_list if edge[1] == target_id]
            top_k_hits = [kinase_idx in sorted_indices[:top_k] for kinase_idx in correct_kinase_idx]

            # Calculate top-K hit for this target and record result
            valid_pre_result_list.append(top_k_hits)

    # Finalizing metrics and plotting
    valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
    valid_label_list = torch.cat(valid_label_list, dim=0)
    true_prob_list = torch.cat(true_prob_list, dim=0)
    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

    metrics.show_result()
    logging.info('Final test; recall: {}, precision: {}, F1: {}, AUROC: {} AUPRC: {}'.format(metrics.Recall, metrics.Precision,
        metrics.F1, metrics.Auc, metrics.Aupr))
    logging.info(f'Final test: {valid_pre_result_list}')
    logging.info(f'Final test: {valid_label_list}')

    # Convert tensors to numpy arrays for use with sklearn
    true_labels = valid_label_list.numpy()
    predicted_probs = true_prob_list.numpy()

    # Calculate ROC curve and AUC (Area Under the Curve)
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall curve and AUPRC (Area Under Precision-Recall Curve)
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
    auprc = average_precision_score(true_labels, predicted_probs)
    confusion = confusion_matrix(true_labels, valid_pre_result_list.numpy())
    
    # Plot the ROC curve
    plt.figure(figsize=(10, 5))

    # ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.5f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random model
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUPRC = {auprc:.5f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")

    # Show both plots
    plt.tight_layout()
    plt.savefig(f'retest_sequence_khop_results.png')
    plt.show()

    # Print the calculated metrics
    logging.info(f"ROC AUC: {roc_auc:.5f}")
    logging.info(f"AUPRC: {auprc:.5f}")
    logging.info(f"Confusion matrix: {confusion}")




def main():
    text_trap = io.StringIO()
    sys.stdout = text_trap
    logging.info("started khop")
    ppi_data = GNN_DATA("balanced_dataset.csv", config, data_folder)
    logging.info("initialized")
    ppi_data.get_feature_origin()
    logging.info("protein features")
    ppi_data.generate_data()
    logging.info("data generated")
    ppi_data.split_dataset(train_valid_index_path=f'{data_folder}/train_val_test_split_1.json', random_new=True,
                           mode='random')
    logging.info("dataset split")
    graph = ppi_data.data
    ppi_list = ppi_data.ppi_list

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']
    graph.test_mask = ppi_data.ppi_split_dict['test_index']

    folder_path = f"{data_folder}"

    p_x_all, p_edge_all = get_features_and_edgelist_all()

    logging.info("big graph loaded")
    p_x_all, x_num_index, num_protein = multi2big_x(p_x_all)
    logging.info("first big")
    p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)
    logging.info("second big")

    batch = multi2big_batch(x_num_index, num_protein)+1



    logging.info("train gnn, train_num: {}, valid_num: {}, test_num: {}".format(len(graph.train_mask), len(graph.val_mask), len(graph.test_mask)))

    graph.edge_index_got = torch.cat(
        (graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    graph.edge_attr_got = torch.cat((graph.edge_attr_1[graph.train_mask], graph.edge_attr_1[graph.train_mask]), dim=0)
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    logging.info(device)

    graph.to(device)

    model = ppi_model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    save_path = f'{data_folder}_result'
    pos_weight = torch.tensor([2.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weight).to(device)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    time_stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    save_path = os.path.join(save_path, "gnn_{}".format('training_seed_1'))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")

    summary_writer = SummaryWriter(save_path)

    train(batch, p_x_all, p_edge_all, model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=1100, epochs=30, scheduler=scheduler,
          got=True)

    summary_writer.close()
    checkpoint = torch.load(os.path.join(save_path, 'gnn_model_valid_best.ckpt'))
    model.load_state_dict(checkpoint['state_dict'])
    test(ppi_data,model, graph, graph.test_mask, device,batch, p_x_all, p_edge_all)


if __name__ == "__main__":
    main()