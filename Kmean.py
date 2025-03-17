import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('processed_netflix_trailers.csv')

# Step 1: Preprocess data (only focus on 'Like Rate' and 'Category')
df['Category'] = df['Category'].astype(str)
df = df.dropna(subset=['Like Rate', 'Category'])

# Step 2: Normalize 'Like Rate' for better clustering performance
scaler = StandardScaler()
df['Normalized Like Rate'] = scaler.fit_transform(df[['Like Rate']])

# Step 3: Apply KMeans clustering to 'Like Rate'
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Normalized Like Rate']])

# Step 4: Calculate the category distribution in each cluster
category_distribution = df.groupby(['Cluster', 'Category']).size().unstack(fill_value=0)

# Step 5: Calculate the overall category distribution
overall_category_distribution = df['Category'].value_counts(normalize=True) * 100

# Step 6: Extract the most liked and least liked clusters (Cluster 0 will be the most liked, Cluster 2 the least liked)
most_liked_cluster = category_distribution.loc[category_distribution.sum(axis=1).idxmax()]
least_liked_cluster = category_distribution.loc[category_distribution.sum(axis=1).idxmin()]

# Step 7: Sort categories by their overall frequency (from most to least frequent)
sorted_categories = overall_category_distribution.sort_values(ascending=False).index

# Step 8: Reorder the clusters to follow the sorted categories
most_liked_cluster_percentage = most_liked_cluster[sorted_categories] / most_liked_cluster.sum() * 100
least_liked_cluster_percentage = least_liked_cluster[sorted_categories] / least_liked_cluster.sum() * 100
overall_category_distribution_sorted = overall_category_distribution[sorted_categories]

# Step 9: Plot the category distributions in the most liked and least liked clusters and overall
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the category distribution in the most liked cluster
ax.plot(most_liked_cluster_percentage.index, most_liked_cluster_percentage.values, label='Most Liked Cluster', marker='o')

# Plot the category distribution in the least liked cluster
ax.plot(least_liked_cluster_percentage.index, least_liked_cluster_percentage.values, label='Least Liked Cluster', marker='o')

# Plot the overall category distribution
ax.plot(overall_category_distribution_sorted.index, overall_category_distribution_sorted.values, label='Overall Distribution', marker='o', linestyle='--')

# Formatting the plot
ax.set_xlabel('Category')
ax.set_ylabel('Percentage (%)')
ax.set_title('Category Distribution in Most and Least Liked Clusters vs Overall')
ax.legend()

# Show the plot
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
