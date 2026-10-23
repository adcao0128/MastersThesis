import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(filename='logs/test.log', encoding='utf-8', level=logging.INFO)

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

def create_negatives():
    # Load your dataset (positive samples)
    file_path = 'positive_associations.csv'  # Replace with your actual file path
    df = pd.read_csv(file_path)

    # Create an empty list to store hard negative pairs
    hard_negatives = []
    all_kinases = df['KIN_ACC_ID'].unique()

    # Total number of rows to track progress
    total_rows = len(df)

    # Iterate over each row in the positive samples
    for idx, row in enumerate(df.iterrows()):
        row = row[1]  # Get the actual row data from the iterator
        
        substrate_id = row['SUB_ACC_ID']
        substrate_seq = row['SITE_+/-7_AA']
        
        # Get the list of kinases that are associated with this substrate in the positive dataset
        positive_kinases = df[df['SITE_+/-7_AA'] == substrate_seq]['KIN_ACC_ID'].unique()

        # Generate negative samples by associating the substrate with kinases not already in the positive samples
        # Get all kinases in the dataset
        all_kinases = df['KIN_ACC_ID'].unique()

        # Exclude kinases that are already paired with the substrate (positive kinases)
        negative_kinases = [kinase for kinase in all_kinases if kinase not in positive_kinases]

        # Create negative pairs by pairing the substrate with each kinase not already associated with it
        for kinase in negative_kinases:
            hard_negatives.append({
                'KIN_ACC_ID': kinase,  # Assuming the kinase ID is available in the row
                'SUB_ACC_ID': substrate_id,
                'SITE_+/-7_AA': substrate_seq,
            })

        # Print progress every 1000 rows
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{total_rows} rows")

    # Convert the list of hard negatives into a DataFrame
    hard_negatives_df = pd.DataFrame(hard_negatives)

    # Optionally, save the hard negatives to a CSV file
    hard_negatives_df.to_csv('hard_negatives.csv', index=False)

    # Preview the first few rows of hard negatives
    print(hard_negatives_df.head())

def create_balanced_dataset():
    positive_df = pd.read_csv("positive_associations.csv")
    positive_df = positive_df[['KIN_ACC_ID', 'SUB_ACC_ID', 'SITE_+/-7_AA']]

    positive_df['true_label'] = 1

    print(positive_df.head())

    hard_negatives_df = pd.read_csv('hard_negatives.csv')
    hard_negatives_df["true_label"] = 0

    print(hard_negatives_df.head())

    neg_grouped = hard_negatives_df.groupby('SITE_+/-7_AA')
    positive_grouped = positive_df.groupby('SITE_+/-7_AA')

    # Initialize an empty list to store sampled negatives and positives
    sampled_negatives = []
    all_positives = []

    # Iterate over positive groups
    for site, pos_group in positive_grouped:
        # Get the corresponding negative group for the same site
        neg_group = neg_grouped.get_group(site) if site in neg_grouped.groups else None
        
        if neg_group is not None:
            # Sample from negatives, same size as the positives group
            sampled_neg = neg_group.sample(n=2 * len(pos_group), random_state=42, replace=False)
            sampled_negatives.append(sampled_neg)
        else:
            print("Site not found")
        
        # Collect the positive group
        all_positives.append(pos_group)

    # Combine all positives and sampled negatives into a single DataFrame
    balanced_negatives_df = pd.concat(sampled_negatives, ignore_index=True)
    balanced_positives_df = pd.concat(all_positives, ignore_index=True)

    # Combine positives and negatives into the final dataset
    balanced_dataset = pd.concat([balanced_positives_df, balanced_negatives_df], ignore_index=True)

    print(balanced_dataset)

    balanced_dataset.to_csv('balanced_dataset.csv', index=False)


if __name__ == "__main__":
    df = read_psp_dataset()
    df = clean_txt_from_pdb(df)

    df.to_csv("positive_associations.csv", index = False)
    create_negatives()
    create_balanced_dataset()