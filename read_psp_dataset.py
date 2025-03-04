import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import logging
logging.basicConfig(filename='./logs/alphafold_query.log', encoding='utf-8', level=logging.INFO)

def read_psp_dataset(filepath="Kinase_Substrate_Dataset.txt"):
    df = pd.read_csv(filepath, sep='\t')
    df = df[["KIN_ACC_ID", "KIN_ORGANISM", "SUB_ACC_ID", "SITE_+/-7_AA"]]
    df = df[df["KIN_ORGANISM"] == "human"]

    return df

def get_uq_acc_ids(df):
    uq_acc_ids = pd.unique(df[['KIN_ACC_ID', 'SUB_ACC_ID']].values.ravel())
    return uq_acc_ids

def get_protein(uniprot_accession):
    api_endpoint = "https://alphafold.ebi.ac.uk/api/prediction/"
    url = f"{api_endpoint}{uniprot_accession}"  # Construct the URL for API

    try:
        # Use a timeout to handle potential connection issues
        response = requests.get(url, timeout=10)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            # Raise an exception for better error handling
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.info(f"Error: {e}")


def download_single_pdb(acc_id, folder_path="./alphafold_pdb_files"):
    protein_info = get_protein(acc_id)
    if protein_info:
        # Extract PDB and PNG URLs
        pdb_url = protein_info[0].get('pdbUrl')

        if pdb_url:
            pdb_data = requests.get(pdb_url).content
            with open(f"{folder_path}/{acc_id}.pdb", mode="wb") as file:
                file.write(pdb_data)
        else:
            logging.info("Failed to retrieve PDB URLs.")
    else:
        logging.info("Failed to retrieve protein information.")

def download_pdb_files(uq_acc_ids):
    MAX_THREADS = 8

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        result = list(executor.map(download_single_pdb, uq_acc_ids))
    

if __name__ == "__main__":
    df = read_psp_dataset()
    uq_acc_ids = get_uq_acc_ids(df)
    print(len(uq_acc_ids))
    # download_pdb_files(uq_acc_ids)