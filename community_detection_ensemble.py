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

def get_ensemble_partitions(graph, resolutions=None, n_runs=20):
    if resolutions is None:
        resolutions = [0.95, 1.0, 1.05]  # Reduced resolution range
    
    partitions = []
    modularities = []
    
    for res in resolutions:
        for _ in range(n_runs):
            partition = best_partition(graph, resolution=res, random_state=None)
            mod = modularity(partition, graph)
            partitions.append(partition)
            modularities.append(mod)
    
    # Select top 3 partitions based on modularity
    top_indices = np.argsort(modularities)[-3:]
    return [partitions[i] for i in top_indices]

def merge_small_communities(partition, graph, min_size=4):
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
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        degree_cent = nx.degree_centrality(G)
        
        importance = {}
        for node in G.nodes():
            importance[node] = (0.6 * pagerank.get(node, 0) + 
                              0.4 * degree_cent.get(node, 0))
        return importance
    except:
        return nx.degree_centrality(G)

def calculate_node_similarity(G, node_pairs):
    all_nodes = set()
    for node1, node2 in node_pairs:
        all_nodes.add(node1)
        all_nodes.add(node2)
    
    # Pre-compute all necessary metrics
    neighbor_sets = {node: set(G.neighbors(node)) for node in all_nodes}
    node_degrees = {node: len(neighbors) for node, neighbors in neighbor_sets.items()}
    
    try:
        clustering_coeffs = nx.clustering(G, nodes=all_nodes)
    except:
        clustering_coeffs = {node: 0 for node in all_nodes}
    
    similarities = np.zeros(len(node_pairs))
    
    for i, (node1, node2) in enumerate(node_pairs):
        try:
            neighbors1 = neighbor_sets[node1]
            neighbors2 = neighbor_sets[node2]
            
            if not neighbors1 or not neighbors2:
                continue
            
            # Basic metrics
            intersection = len(neighbors1 & neighbors2)
            union = len(neighbors1 | neighbors2)
            jaccard = intersection / union if union > 0 else 0
            
            common_neighbors = intersection
            max_neighbors = max(len(neighbors1), len(neighbors2))
            common_ratio = common_neighbors / max_neighbors if max_neighbors > 0 else 0
            
            # Advanced metrics
            resource_alloc = sum(1 / (node_degrees[n] + 1) for n in neighbors1 & neighbors2)
            pref_attach = (node_degrees[node1] * node_degrees[node2]) / (len(G.nodes()) ** 2)
            
            # Structural metrics
            clust_sim = 1 - abs(clustering_coeffs.get(node1, 0) - clustering_coeffs.get(node2, 0))
            
            # Optimized weights based on feature importance
            similarities[i] = (0.35 * jaccard + 
                             0.30 * common_ratio + 
                             0.20 * resource_alloc + 
                             0.10 * pref_attach +
                             0.05 * clust_sim)
            
        except:
            continue
    
    return similarities

def predict_community_membership(G, node_pairs, partitions, similarities):
    predictions = np.zeros(len(node_pairs), dtype=np.int32)
    node_importance = calculate_node_importance(G)
    
    # Calculate community sizes for each partition
    community_sizes = []
    for partition in partitions:
        sizes = defaultdict(int)
        for node, comm_id in partition.items():
            sizes[comm_id] += 1
        community_sizes.append(sizes)
    
    # Pre-compute node degrees
    node_degrees = dict(G.degree())
    
    for i, (node1, node2) in enumerate(node_pairs):
        # Get ensemble prediction
        ensemble_pred = 0
        for partition, sizes in zip(partitions, community_sizes):
            if node1 in partition and node2 in partition:
                if partition[node1] == partition[node2]:
                    ensemble_pred += 1
                    
                    # Adjust based on community sizes
                    comm1_size = sizes[partition[node1]]
                    comm2_size = sizes[partition[node2]]
                    size_factor = min(comm1_size, comm2_size) / max(comm1_size, comm2_size)
                    if size_factor < 0.3:  # If communities are very different in size
                        ensemble_pred -= 0.5
        
        # Normalize ensemble prediction
        ensemble_pred = ensemble_pred / len(partitions)
        
        # Base prediction
        predictions[i] = 1 if ensemble_pred > 0.5 else 0
        
        # If not in same community, check additional features
        if predictions[i] == 0:
            try:
                # Path length with importance weighting
                path_length = nx.shortest_path_length(G, node1, node2)
                importance_score = (node_importance.get(node1, 0) + node_importance.get(node2, 0)) / 2
                
                # Dynamic threshold based on node properties
                base_threshold = 0.20
                threshold = base_threshold - 0.05 * importance_score
                
                if path_length <= 2 and similarities[i] > threshold:
                    predictions[i] = 1
                
                # Check local structure
                neighbors1 = set(G.neighbors(node1))
                neighbors2 = set(G.neighbors(node2))
                common_neighbors = neighbors1 & neighbors2
                
                if len(common_neighbors) >= 2:
                    # Check if common neighbors are in same community
                    common_community = all(any(p[n] == p[list(common_neighbors)[0]] 
                                            for p in partitions)
                                        for n in common_neighbors)
                    if common_community:
                        predictions[i] = 1
                    
                    # Check if nodes have high similarity
                    if similarities[i] > (0.30 - 0.05 * importance_score):
                        predictions[i] = 1
                    
                    # Check if nodes have strong local connections
                    if len(common_neighbors) >= 3 and similarities[i] > 0.25:
                        predictions[i] = 1
                    
                    # Check degree similarity
                    degree_sim = 1 - abs(node_degrees[node1] - node_degrees[node2]) / max(node_degrees[node1], node_degrees[node2])
                    if degree_sim > 0.75 and similarities[i] > 0.22:
                        predictions[i] = 1
            except:
                pass
        else:
            # Additional checks for positive predictions
            try:
                if similarities[i] < 0.15:  # If similarity is very low
                    predictions[i] = 0
            except:
                pass
    
    return predictions

def main():
    train_file = r"C:\Users\Dspike\Documents\NTUST\SNet\train.csv"
    test_file = r"C:\Users\Dspike\Documents\NTUST\SNet\test.csv"
    output_file = r"C:\Users\Dspike\Documents\NTUST\SNet\sample_submission.csv"
    
    print("Loading data...")
    train_df, test_df = load_data(train_file, test_file)
    
    print("Creating graph...")
    G = create_graph(train_df)
    
    print("Finding ensemble of community partitions...")
    partitions = get_ensemble_partitions(G)
    partitions = [merge_small_communities(p, G, min_size=4) for p in partitions]
    
    print("Calculating node similarities...")
    node_pairs = list(zip(test_df['Node1'], test_df['Node2']))
    similarities = calculate_node_similarity(G, node_pairs)
    
    print("Making predictions...")
    predictions = predict_community_membership(G, node_pairs, partitions, similarities)
    
    print("Creating submission file...")
    submission_df = pd.DataFrame({
        'Id': range(len(predictions)),
        'Category': predictions
    })
    submission_df = submission_df[['Id', 'Category']]
    submission_df.to_csv(output_file, index=False)
    
    print(f"Submission file saved to {output_file}")

if __name__ == "__main__":
    main() 