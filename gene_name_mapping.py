# import pandas as pd

# # Load datasets
# df = pd.read_csv("PSPA_ST.csv")
# df2 = pd.read_csv('Kinase_Substrate_Dataset.txt', sep='\t')

# # Filter for human kinases (assuming 'Homo sapiens' is the label)
# df_human = df2[df2['KIN_ORGANISM'] == 'human']

# # Now create the kinase to accession dictionary (one-to-one, last mapping kept)
# kinase_to_acc = dict(df_human[['KINASE', 'KIN_ACC_ID']].drop_duplicates().values)

# # Extract kinase names from _percentile columns
# percentile_cols = [col for col in df.columns if col.endswith('_percentile')]
# kinases_in_df = [col.replace('_percentile', '') for col in percentile_cols]

# # Rebuild new mapping DataFrame
# mapped_rows = []
# unmapped_kinases = []

# for kinase in kinases_in_df:
#     if kinase in kinase_to_acc:
#         mapped_rows.append((kinase, kinase_to_acc[kinase]))
#     else:
#         unmapped_kinases.append(kinase)

# df_new = pd.DataFrame(mapped_rows, columns=['KINASE', 'KIN_ACC_ID'])

# # Save to file (optional)
# df_new.to_csv("new_mapped_kinases.csv", index=False)

# # Print summary
# print(f"✅ Mapped: {len(df_new)}")
# print(f"❌ Unmapped: {len(unmapped_kinases)}")
# if unmapped_kinases:
#     print("Unmapped kinases:", unmapped_kinases)

# # Save kinases to a comma-separated text file
# with open("unmapped_kinases_list.txt", "w") as f:
#     f.write(",".join(unmapped_kinases))

import pandas as pd

# Load main mapping table
df = pd.read_csv("mapping.txt", sep='\t')

# Load filter list from file (comma-separated)
with open("kinases_list.txt", 'r') as f:
    filter_values = [v.strip() for v in f.read().strip().split(',')]

# Prepare set of all possible symbols in the mapping file
symbol_to_row = {}

for idx, row in df.iterrows():
    approved = str(row['Approved symbol']).strip()
    aliases = str(row['Alias symbols']).split(',') if pd.notna(row['Alias symbols']) else []

    # Add approved symbol
    symbol_to_row[approved] = row

    # Add each alias
    for alias in aliases:
        alias = alias.strip()
        if alias and alias not in symbol_to_row:
            symbol_to_row[alias] = row

# Filter rows where filter value matches approved or alias
matched_rows = []
not_found = []

matched_rows = []
not_found = []

for val in filter_values:
    if val in symbol_to_row:
        row = symbol_to_row[val]
        matched_rows.append({
            'filter_symbol': val,
            'uniprot_id': row['UniProt ID(supplied by UniProt)']
        })
    else:
        not_found.append(val)

# Create DataFrame from matched rows
filtered_df = pd.DataFrame(matched_rows)

# Create DataFrame for not found values with empty uniprot_id
not_found_df = pd.DataFrame({
    'filter_symbol': not_found,
    'uniprot_id': [''] * len(not_found)
})

# Combine both
final_df = pd.concat([filtered_df, not_found_df], ignore_index=True)

# Save to CSV
final_df.to_csv("newest_newest_mapping.csv", index=False)


# Output
print("✅ Matched rows:", len(filtered_df))
print("❌ Not found:", not_found)
print(len(not_found))

print(filtered_df.head())
