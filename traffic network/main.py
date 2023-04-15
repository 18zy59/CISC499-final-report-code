import networkx as nx
import community as community_louvain
import leidenalg
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import osmnx as ox


import geopandas as gpd
from shapely.geometry import Point
# toronto
place_name = "Toronto, Canada"
graph = ox.graph_from_place(place_name, network_type='drive')

# to NetworkX 


place_name = "Toronto, Ontario, Canada"
graph = ox.graph_from_place(place_name, network_type='drive')
G = graph
G = nx.to_undirected(G)  # Convert the graph to an undirected graph
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
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax1)  
    ax1.set_title(louvain_title)
    ax1.axis("off")

    # Leiden
    for community, color in zip(set(leiden_partition.values()), community_colors):
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in leiden_partition if leiden_partition[node] == community],
                               node_color=[color], node_size=100, alpha=0.8, ax=ax2)
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.5, ax=ax2)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax2) 
    ax1.set_title(louvain_title)
    ax2.set_title(leiden_title)
    ax2.axis("off")

    plt.show()

def community_stats(partition):
    community_count = len(set(partition.values()))
    community_sizes = [list(partition.values()).count(i) for i in range(community_count)]
    avg_community_size = sum(community_sizes) / community_count
    return community_count, avg_community_size, community_sizes



# 
visualize_communities_comparison(G, partition_louvain, leiden_partition_dict, 'Louvain Community Detection', 'Leiden Community Detection')


louvain_community_count, louvain_avg_community_size, louvain_community_sizes = community_stats(partition_louvain)
leiden_community_count, leiden_avg_community_size, leiden_community_sizes = community_stats(leiden_partition_dict)

louvain_modularity = community_louvain.modularity(partition_louvain, G)
leiden_modularity = partition_leiden.modularity

print(f"Louvain: {louvain_community_count} communities, average size: {louvain_avg_community_size:.2f}, modularity: {louvain_modularity:.4f}")
print(f"Leiden: {leiden_community_count} communities, average size: {leiden_avg_community_size:.2f}, modularity: {leiden_modularity:.4f}")

# convert to DataFrame
louvain_df = pd.DataFrame(list(partition_louvain.items()), columns=['Node', 'Louvain_Community'])
leiden_df = pd.DataFrame(list(leiden_partition_dict.items()), columns=['Node', 'Leiden_Community'])


merged_df = louvain_df.copy()
merged_df['Leiden_Community'] = leiden_df['Leiden_Community']
merged_df = pd.melt(merged_df, id_vars=['Node'], var_name='Method', value_name='Community')

def visualize_combined_data(data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='Community', hue='Method', data=data, palette="viridis")
    ax.set_title('Louvain and Leiden Community Distribution')
    plt.show()

#  seaborn 
visualize_combined_data(merged_df)
# 
output_file = "C:/Users/Bill/Desktop/499traffic/community_results.xlsx"
writer = pd.ExcelWriter(output_file, engine='openpyxl')

louvain_df.to_excel(writer, sheet_name='Louvain', index=False)
leiden_df.to_excel(writer, sheet_name='Leiden', index=False)


merged_df.to_excel(writer, sheet_name='Combined', index=False)


writer.save()
print(f"Results saved to {output_file}")


location = 'Toronto, Ontario, Canada'
graph = ox.graph_from_place(location, network_type='drive', simplify=True)


nodes, _ = ox.graph_to_gdfs(graph)

louvain_df = pd.read_excel('C:/Users/Bill/Desktop/499traffic/community_results.xlsx', sheet_name='Louvain')
leiden_df = pd.read_excel('C:/Users/Bill/Desktop/499traffic/community_results.xlsx', sheet_name='Leiden')


merged_communities = louvain_df.merge(leiden_df, left_on='Node', right_on='Node', suffixes=('_louvain', '_leiden'))


nodes = nodes.merge(merged_communities, left_on='osmid', right_on='Node')



output_file_path = 'C:/Users/Bill/Desktop/499traffic/community_results.geojson'
nodes.to_file(output_file_path, driver='GeoJSON')