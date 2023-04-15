import networkx as nx
import community as community_louvain
import leidenalg
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sns
from sklearn.cluster import SpectralClustering

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist



with open("dolphins.csv", "r") as input_file:
    csv_reader = csv.reader(input_file, delimiter=',')
    next(csv_reader)  
    edgelist = [",".join(row) for row in csv_reader]
    G = nx.parse_edgelist(edgelist, delimiter=",", nodetype=str)


# Louvain
partition_louvain = community_louvain.best_partition(G)

# Leiden
igraph_graph = ig.Graph.from_networkx(G)
partition_leiden = leidenalg.find_partition(igraph_graph, leidenalg.ModularityVertexPartition)
leiden_partition_dict = {node: membership for node, membership in zip(G.nodes(), partition_leiden.membership)}
def get_spectral_clustering_partition(graph, n_clusters):
    adjacency_matrix = nx.to_numpy_array(graph)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = clustering.fit_predict(adjacency_matrix)
    partition = {node: label for node, label in zip(graph.nodes(), labels)}
    return partition

k = 4  
spectral_partition = get_spectral_clustering_partition(G, k)


def visualize_communities_comparison(graph, louvain_partition, leiden_partition, spectral_partition, louvain_title, leiden_title, spectral_title):
    pos = nx.spring_layout(graph, seed=42)
    community_colors = plt.cm.rainbow(np.linspace(0, 1, max(len(set(louvain_partition.values())), len(set(leiden_partition.values())), len(set(spectral_partition.values())))))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

    # Louvain
    for community, color in zip(set(louvain_partition.values()), community_colors):
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in louvain_partition if louvain_partition[node] == community],
                               node_color=[color], node_size=100, alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.5, ax=ax1)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax1)
    ax1.set_title(louvain_title)
    ax1.axis("off")

    # Leiden
    for community, color in zip(set(leiden_partition.values()), community_colors):
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in leiden_partition if leiden_partition[node] == community],
                               node_color=[color], node_size=100, alpha=0.8, ax=ax2)
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.5, ax=ax2)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax2)
    ax2.set_title(leiden_title)
    ax2.axis("off")

    # Spectral Clustering
    for community, color in zip(set(spectral_partition.values()), community_colors):
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in spectral_partition if spectral_partition[node] == community],
                               node_color=[color], node_size=100, alpha=0.8, ax=ax3)
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.5, ax=ax3)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax3)
    ax3.set_title(spectral_title)
    ax3.axis("off")

    plt.show()

visualize_communities_comparison(G, partition_louvain, leiden_partition_dict, spectral_partition, 'Louvain Community Detection', 'Leiden Community Detection', 'Spectral Clustering (K-means)')

def community_stats(partition):
    community_count = len(set(partition.values()))
    community_sizes = [list(partition.values()).count(i) for i in range(community_count)]
    avg_community_size = sum(community_sizes) / community_count
    return community_count, avg_community_size, community_sizes

louvain_community_count, louvain_avg_community_size, louvain_community_sizes = community_stats(partition_louvain)
leiden_community_count, leiden_avg_community_size, leiden_community_sizes = community_stats(leiden_partition_dict)

louvain_modularity = community_louvain.modularity(partition_louvain, G)
leiden_modularity = partition_leiden.modularity

print(f"Louvain: {louvain_community_count} communities, average size: {louvain_avg_community_size:.2f}, modularity: {louvain_modularity:.4f}")
print(f"Leiden: {leiden_community_count} communities, average size: {leiden_avg_community_size:.2f}, modularity: {leiden_modularity:.4f}")

louvain_df = pd.DataFrame(list(partition_louvain.items()), columns=['Dolphin', 'Louvain_Community'])
leiden_df = pd.DataFrame(list(leiden_partition_dict.items()), columns=['Dolphin', 'Leiden_Community'])




merged_df = louvain_df.copy()
merged_df['Leiden_Community'] = leiden_df['Leiden_Community']
merged_df = pd.melt(merged_df, id_vars=['Dolphin'], var_name='Method', value_name='Community')

#use seaborn
def visualize_combined_data(data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='Community', hue='Method', data=data, palette="viridis")
    ax.set_title('Louvain and Leiden Community Distribution')
    plt.show()


visualize_combined_data(merged_df)


########  K -means part
