import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# Initialize an empty list to store the extracted data
data = []

# Path to your .log file
log_file = './logs/2hop_stats.log'

# Regular expression pattern to extract the ID, existing, adjacent, and total edges
pattern = r"Site Sequence: (\S+), Structure Edges: (\d+), Total Nodes: (\d+)"

# Read the file line by line
with open(log_file, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            # Extract the data from the matched groups and append to the list
            site_sequence, structure_edges, total_edges = match.groups()
            data.append({
                'Site Sequence': site_sequence,
                'Structure Edges': int(structure_edges),
                'Total Edges': int(total_edges)
            })

# Create a pandas DataFrame from the extracted data
df = pd.DataFrame(data)

print(df['Structure Edges'].describe())

# Plot histogram
plt.hist(df['Structure Edges'], bins=105, edgecolor='black')

# Set x-axis ticks every 5 units
plt.xticks(np.arange(0, 105, 5))  # Adjust range as needed

plt.xlabel('# of Edges')
plt.ylabel('Frequency')
plt.title('2-hop Edges')

plt.savefig("2hop_histogram.png")


