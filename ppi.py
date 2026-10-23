import os
import pandas as pd
import h5py

def get_pairs_from_txt(txt_filename):
    df = pd.read_csv(txt_filename, sep='\t')
    df = df[['KIN_ACC_ID', 'SUB_ACC_ID', 'SITE_+/-7_AA']]
    return df

pairs_df = get_unique_pairs_from_hdf5('../graph_data.h5')
txt_df = get_pairs_from_txt('Kinase_Substrate_Dataset.txt')

# Merge the two DataFrames on KIN_ACC_ID and SUB_ACC_ID
merged_df = pd.merge(pairs_df, txt_df[['KIN_ACC_ID', 'SUB_ACC_ID', 'SITE_+/-7_AA']], 
                     on=['KIN_ACC_ID', 'SUB_ACC_ID'], how='left')

merged_df['true_label'] = 1

print(merged_df.head())

hard_negatives_df = pd.read_csv('../hard_negatives_sequence.csv')
hard_negatives_df["true_label"] = 0

print(hard_negatives_df.head())

neg_grouped = hard_negatives_df.groupby('SITE_+/-7_AA')
merged_grouped = merged_df.groupby('SITE_+/-7_AA')

# Initialize an empty list to store sampled negatives and positives
sampled_negatives = []
all_positives = []

# Iterate over positive groups
for site, pos_group in merged_grouped:
    # Get the corresponding negative group for the same site
    neg_group = neg_grouped.get_group(site) if site in neg_grouped.groups else None
    
    if neg_group is not None:
        # Sample from negatives, same size as the positives group
        sampled_neg = neg_group.sample(n=len(pos_group), random_state=42, replace=False)
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