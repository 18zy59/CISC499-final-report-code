import networkx as nx
import community as community_louvain
import leidenalg
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sns
import geopandas as gpd

import time

with open("diplomatic_relations.csv", "r") as input_file:
    csv_reader = csv.reader(input_file, delimiter=',')
    next(csv_reader)  # Skip the first row (header row)
    edgelist = [",".join(row) for row in csv_reader]
    G = nx.parse_edgelist(edgelist, delimiter=",", nodetype=str)

# Louvain
partition_louvain = community_louvain.best_partition(G)

# Leiden
igraph_graph = ig.Graph.from_networkx(G)
partition_leiden = leidenalg.find_partition(igraph_graph, leidenalg.ModularityVertexPartition)
leiden_partition_dict = {node: membership for node, membership in zip(G.nodes(), partition_leiden.membership)}

def visualize_communities_comparison(graph, louvain_partition, leiden_partition, louvain_title, leiden_title):
    pos = nx.spring_layout(graph, seed=42)
    community_colors = plt.cm.rainbow(np.linspace(0, 1, max(len(set(louvain_partition.values())), len(set(leiden_partition.values())))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Louvain
    for community, color in zip(set(louvain_partition.values()), community_colors):
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in louvain_partition if louvain_partition[node] == community],
                               node_color=[color], node_size=100, alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.5, ax=ax1)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax1)  # Add country labels
    ax1.set_title(louvain_title)
    ax1.axis("off")

    # Leiden
    for community, color in zip(set(leiden_partition.values()), community_colors):
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in leiden_partition if leiden_partition[node] == community],
                               node_color=[color], node_size=100, alpha=0.8, ax=ax2)
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.5, ax=ax2)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax2)  # Add country labels
    ax2.set_title(leiden_title)
    ax2.axis("off")

    plt.show()

# Visualize comparison
visualize_communities_comparison(G, partition_louvain, leiden_partition_dict, 'Louvain Community Detection', 'Leiden Community Detection')


def community_stats(partition):
    community_count = len(set(partition.values()))
    community_sizes = [list(partition.values()).count(i) for i in range(community_count)]
    avg_community_size = sum(community_sizes) / community_count
    return community_count, avg_community_size, community_sizes

louvain_community_count, louvain_avg_community_size, louvain_community_sizes = community_stats(partition_louvain)
leiden_community_count, leiden_avg_community_size, leiden_community_sizes = community_stats(leiden_partition_dict)




louvain_df = pd.DataFrame(list(partition_louvain.items()), columns=['Country', 'Louvain_Community'])

# create Leiden DataFrame
leiden_df = pd.DataFrame(list(leiden_partition_dict.items()), columns=['Country', 'Leiden_Community'])


merged_df = louvain_df.copy()
merged_df['Leiden_Community'] = leiden_df['Leiden_Community']
merged_df = pd.melt(merged_df, id_vars=['Country'], var_name='Method', value_name='Community')

def visualize_combined_data(data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='Community', hue='Method', data=data, palette="viridis")
    ax.set_title('Louvain and Leiden Community Distribution')
    plt.show()


visualize_combined_data(merged_df)


louvain_df = pd.DataFrame(list(partition_louvain.items()), columns=['Country', 'Louvain_Community'])
leiden_df = pd.DataFrame(list(leiden_partition_dict.items()), columns=['Country', 'Leiden_Community'])

# Merge Louvain and Leiden results
merged_df = louvain_df.copy()
merged_df['Leiden_Community'] = leiden_df['Leiden_Community']

# Save the merged results to a CSV file
merged_df.to_csv("C:/Users/Bill/Desktop/499country/community_detection_results.csv", index=False)











louvain_modularity = community_louvain.modularity(partition_louvain, G)


leiden_modularity = leidenalg.ModularityVertexPartition(igraph_graph).quality()


start_time_louvain = time.time()
community_louvain.best_partition(G)
end_time_louvain = time.time()
runtime_louvain = end_time_louvain - start_time_louvain


start_time_leiden = time.time()
leidenalg.find_partition(igraph_graph, leidenalg.ModularityVertexPartition)
end_time_leiden = time.time()
runtime_leiden = end_time_leiden - start_time_leiden


print(f"Louvain Modularity: {louvain_modularity:.4f}")
print(f"Leiden Modularity: {leiden_modularity:.4f}")
print(f"Louvain Runtime: {runtime_louvain:.2f} seconds")
print(f"Leiden Runtime: {runtime_leiden:.2f} seconds")








