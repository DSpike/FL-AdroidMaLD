import pandas as pd
import networkx as nx
from community.community_louvain import best_partition, modularity
import numpy as np
from collections import defaultdict
import warnings
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df

def create_graph(train_df):
    G = nx.Graph()
    edges = list(zip(train_df['Node1'], train_df['Node2']))
    G.add_edges_from(edges)
    return G

def get_best_partition(graph, resolution=1.0, n_runs=21):
    best_part = None
    best_mod = -float('inf')
    
    for _ in range(n_runs):
        partition = best_partition(graph, resolution=resolution, random_state=None)
        mod = modularity(partition, graph)
        if mod > best_mod:
            best_mod = mod
            best_part = partition
    return best_part, best_mod

def merge_small_communities(partition, graph, min_size=5):
    community_sizes = defaultdict(int)
    for node, comm_id in partition.items():
        community_sizes[comm_id] += 1
    
    small_comms = {comm_id for comm_id, size in community_sizes.items() if size < min_size}
    if not small_comms:
        return partition
    
    new_partition = partition.copy()
    for node in graph.nodes():
        if partition[node] in small_comms:
            neighbor_comms = defaultdict(int)
            for neighbor in graph.neighbors(node):
                if partition[neighbor] not in small_comms:
                    neighbor_comms[partition[neighbor]] += 1
            if neighbor_comms:
                new_comm = max(neighbor_comms.items(), key=lambda x: x[1])[0]
                new_partition[node] = new_comm
    
    return new_partition

def calculate_node_importance(G):
    try:
        return nx.pagerank(G)
    except:
        # Fallback to degree centrality if PageRank fails
        return nx.degree_centrality(G)

def calculate_node_similarity(G, node_pairs):
    # Pre-compute neighbor sets and node degrees
    all_nodes = set()
    for node1, node2 in node_pairs:
        all_nodes.add(node1)
        all_nodes.add(node2)
    
    neighbor_sets = {node: set(G.neighbors(node)) for node in all_nodes}
    node_degrees = {node: len(neighbors) for node, neighbors in neighbor_sets.items()}
    
    # Pre-compute clustering coefficients
    try:
        clustering_coeffs = nx.clustering(G, nodes=all_nodes)
    except:
        clustering_coeffs = {node: 0 for node in all_nodes}
    
    # Pre-compute betweenness centrality for important nodes
    try:
        important_nodes = {node for node in all_nodes if node_degrees[node] > np.mean(list(node_degrees.values()))}
        betweenness = nx.betweenness_centrality(G, k=min(1000, len(G.nodes())))
    except:
        betweenness = {node: 0 for node in all_nodes}
    
    similarities = np.zeros(len(node_pairs))
    
    for i, (node1, node2) in enumerate(node_pairs):
        try:
            # Get neighbor sets
            neighbors1 = neighbor_sets[node1]
            neighbors2 = neighbor_sets[node2]
            
            if not neighbors1 or not neighbors2:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(neighbors1 & neighbors2)
            union = len(neighbors1 | neighbors2)
            jaccard = intersection / union if union > 0 else 0
            
            # Calculate common neighbors ratio
            common_neighbors = intersection
            max_neighbors = max(len(neighbors1), len(neighbors2))
            common_ratio = common_neighbors / max_neighbors if max_neighbors > 0 else 0
            
            # Calculate resource allocation index
            common_neighbors_set = neighbors1 & neighbors2
            resource_alloc = sum(1 / (node_degrees[n] + 1) for n in common_neighbors_set)
            
            # Calculate preferential attachment score
            pref_attach = (node_degrees[node1] * node_degrees[node2]) / (len(G.nodes()) ** 2)
            
            # Calculate Adamic-Adar index
            adamic_adar = sum(1 / np.log(node_degrees[n] + 1) for n in common_neighbors_set)
            
            # Calculate clustering coefficient similarity
            clust_sim = 1 - abs(clustering_coeffs.get(node1, 0) - clustering_coeffs.get(node2, 0))
            
            # Calculate betweenness centrality similarity
            between_sim = 1 - abs(betweenness.get(node1, 0) - betweenness.get(node2, 0))
            
            # Calculate local clustering coefficient
            try:
                local_clust1 = nx.clustering(G, nodes=neighbors1)
                local_clust2 = nx.clustering(G, nodes=neighbors2)
                local_clust_sim = 1 - abs(np.mean(list(local_clust1.values())) - np.mean(list(local_clust2.values())))
            except:
                local_clust_sim = 0
            
            # Combine metrics with optimized weights
            similarities[i] = (0.35 * jaccard + 
                             0.25 * common_ratio + 
                             0.15 * resource_alloc + 
                             0.1 * pref_attach + 
                             0.05 * adamic_adar +
                             0.05 * clust_sim +
                             0.03 * between_sim +
                             0.02 * local_clust_sim)
            
        except:
            continue
    
    return similarities

def predict_community_membership(G, node_pairs, best_partition, similarities):
    predictions = np.zeros(len(node_pairs), dtype=np.int32)
    
    # Pre-compute node importance scores
    node_importance = calculate_node_importance(G)
    
    for i, (node1, node2) in enumerate(node_pairs):
        # Check if both nodes exist in the partition
        if node1 in best_partition and node2 in best_partition:
            # Base prediction
            predictions[i] = 1 if best_partition[node1] == best_partition[node2] else 0
            
            # If not in same community, check additional features
            if predictions[i] == 0:
                try:
                    # Check path length with importance weighting
                    path_length = nx.shortest_path_length(G, node1, node2)
                    importance_score = (node_importance.get(node1, 0) + node_importance.get(node2, 0)) / 2
                    
                    if path_length <= 2 and similarities[i] > (0.25 - 0.05 * importance_score):
                        predictions[i] = 1
                except nx.NetworkXNoPath:
                    pass
                
                # Check local structure
                try:
                    neighbors1 = set(G.neighbors(node1))
                    neighbors2 = set(G.neighbors(node2))
                    common_neighbors = neighbors1 & neighbors2
                    
                    if len(common_neighbors) >= 2:
                        # Check if common neighbors are in the same community
                        common_community = all(best_partition[n] == best_partition[list(common_neighbors)[0]] 
                                            for n in common_neighbors)
                        if common_community:
                            predictions[i] = 1
                        
                        # Check if nodes have high similarity with importance weighting
                        similarity_threshold = 0.35 - 0.05 * importance_score
                        if similarities[i] > similarity_threshold:
                            predictions[i] = 1
                            
                        # Check if nodes have strong local connections
                        if len(common_neighbors) >= 3 and similarities[i] > 0.3:
                            predictions[i] = 1
                except:
                    pass
        else:
            # Fallback for nodes not in partition with importance weighting
            importance_score = (node_importance.get(node1, 0) + node_importance.get(node2, 0)) / 2
            if similarities[i] > (0.35 - 0.05 * importance_score):
                predictions[i] = 1
    
    return predictions

def main():
    # File paths
    train_file = r"C:\Users\Dspike\Documents\NTUST\SNet\train.csv"
    test_file = r"C:\Users\Dspike\Documents\NTUST\SNet\test.csv"
    output_file = r"C:\Users\Dspike\Documents\NTUST\SNet\sample_submission.csv"
    
    print("Loading data...")
    train_df, test_df = load_data(train_file, test_file)
    
    print("Creating graph...")
    G = create_graph(train_df)
    
    print("Finding best community partition...")
    resolutions = [0.9, 0.95, 0.96, 1.0, 1.05, 1.1]
    best_overall_partition = None
    best_overall_modularity = -float('inf')
    
    for res in resolutions:
        partition, modularity_score = get_best_partition(G, resolution=res, n_runs=21)
        partition = merge_small_communities(partition, G, min_size=5)
        modularity_score = modularity(partition, G)
        print(f"Resolution {res}: Modularity = {modularity_score}")
        if modularity_score > best_overall_modularity:
            best_overall_modularity = modularity_score
            best_overall_partition = partition
    
    print("Calculating node similarities...")
    node_pairs = list(zip(test_df['Node1'], test_df['Node2']))
    similarities = calculate_node_similarity(G, node_pairs)
    
    print("Making predictions...")
    predictions = predict_community_membership(G, node_pairs, best_overall_partition, similarities)
    
    print("Creating submission file...")
    submission_df = pd.DataFrame({
        'Id': range(len(predictions)),
        'Category': predictions
    })
    submission_df = submission_df[['Id', 'Category']]
    submission_df.to_csv(output_file, index=False)
    
    print(f"Submission file saved to {output_file}")
    print(f"Best modularity achieved: {best_overall_modularity}")

if __name__ == "__main__":
    main() 