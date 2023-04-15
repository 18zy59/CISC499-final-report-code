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