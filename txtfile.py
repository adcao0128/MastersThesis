import pandas as pd

# Load the CSV
df = pd.read_csv("kinase_mapping.csv")

# Extract the 'uniprot_id' column
filter_symbols = df['filter_symbol'].dropna().unique()

print(len(filter_symbols))

# Join into a comma-separated string
id_string = ",".join(filter_symbols)

# Save to TXT
with open("filter_symbols.txt", "w") as f:
    f.write(id_string)
