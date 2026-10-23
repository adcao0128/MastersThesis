import pandas as pd

#df = pd.read_csv('balanced_dataset.csv')

# df_filtered = df[df['true_label'] == 1]
df = pd.read_csv('Kinase_Substrate_Dataset.txt', sep='\t')
df = df[["KIN_ACC_ID", "KIN_ORGANISM", "SUB_ACC_ID", "SITE_+/-7_AA"]]
df_filtered = df[df["KIN_ORGANISM"] == "human"]

kinase_site_counts = (
    df_filtered.groupby('KIN_ACC_ID')['SITE_+/-7_AA']
    .nunique()
    .reset_index(name='site_count')
)

# Step 3: Categorize kinases
def categorize(count):
    if count < 5:
        return 'poor'
    elif count <= 20:
        return 'average'
    else:
        return 'rich'

kinase_site_counts['category'] = kinase_site_counts['site_count'].apply(categorize)

print(kinase_site_counts.head())
# print(kinase_site_counts.info())
# print(kinase_site_counts.describe())

category_counts = kinase_site_counts['category'].value_counts().reindex(['poor', 'average', 'rich'], fill_value=0)

print(category_counts)

import matplotlib.pyplot as plt
import seaborn as sns

bins = list(range(0, 201, 5)) + [float('inf')]
labels = [f"{i}-{i+5}" for i in range(0, 200, 5)] + ['>200']


kinase_site_counts['binned'] = pd.cut(
    kinase_site_counts['site_count'],
    bins=bins,
    labels=labels,
    right=False
)


plt.figure(figsize=(10, 6))
sns.countplot(data=kinase_site_counts, x='binned', order=labels)
plt.title('Distribution of Site Counts per Kinase (Binned by 5, >200)')
plt.xlabel('Site Count Range')
plt.ylabel('Number of Kinases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'kinase_distribution.png')
plt.show()

total_sites = kinase_site_counts['site_count'].sum()


lt_20_sites = kinase_site_counts[kinase_site_counts['site_count'] < 200]['site_count'].sum()
lt_20_pct = (lt_20_sites / total_sites) * 100

gt_200_sites = kinase_site_counts[kinase_site_counts['site_count'] > 200]['site_count'].sum()
gt_200_pct = (gt_200_sites / total_sites) * 100

print(f"Percentage of sites from kinases with <200 sites: {lt_20_pct:.2f}%")
print(f"Percentage of sites from kinases with >200 sites: {gt_200_pct:.2f}%")