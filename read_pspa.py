import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import logging
import os

session = requests.Session()

def read_psp_dataset(filepath="Kinase_Substrate_Dataset.txt"):
    df = pd.read_csv(filepath, sep='\t')
    df = df[["KIN_ACC_ID", "KIN_ORGANISM", "SUB_ACC_ID", "SITE_+/-7_AA"]]
    df = df[df["KIN_ORGANISM"] == "human"]
    return df

def read_pspa_dataset(filepath="PSPA_ST.csv"):
    df = pd.read_csv(filepath)
    return df["Uniprot Primary Accession"]

def get_uq_acc_ids(df_ks, uniprot_list):
    # Combine the three columns and drop duplicates
    combined = pd.concat([
        df_ks['KIN_ACC_ID'],
        df_ks['SUB_ACC_ID'],
        uniprot_list
    ]).dropna().unique()
    return combined

def get_protein(uniprot_accession):
    api_endpoint = "https://alphafold.ebi.ac.uk/api/prediction/"
    url = f"{api_endpoint}{uniprot_accession}"

    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.info(f"Error for {uniprot_accession}: {e}")

def download_single_pdb(acc_id, folder_path="./alphafold_pdb_files"):
    os.makedirs(folder_path, exist_ok=True)
    protein_info = get_protein(acc_id)
    print(acc_id)
    if protein_info:
        pdb_url = protein_info[0].get('pdbUrl')
        if pdb_url:
            pdb_data = session.get(pdb_url).content
            with open(f"{folder_path}/{acc_id}.pdb", mode="wb") as file:
                file.write(pdb_data)
        else:
            logging.info(f"No PDB URL for {acc_id}.")
    else:
        logging.info(f"No protein info for {acc_id}.")

def download_pdb_files(uq_acc_ids):
    MAX_THREADS = 20
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        executor.map(download_single_pdb, uq_acc_ids)

if __name__ == "__main__":
    df_ks = read_psp_dataset()
    uniprot_list = read_pspa_dataset()
    uq_acc_ids = get_uq_acc_ids(df_ks, uniprot_list)
    print(f"Total unique accession IDs: {len(uq_acc_ids)}")
    download_pdb_files(uq_acc_ids)
