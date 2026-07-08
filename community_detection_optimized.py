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

def get_best_partition(graph, resolution=1.0, n_runs=20):
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

def calculate_node_similarity(G, node_pairs):
    # Pre-compute neighbor sets and node degrees
    all_nodes = set()
    for node1, node2 in node_pairs:
        all_nodes.add(node1)
        all_nodes.add(node2)
    
    neighbor_sets = {node: set(G.neighbors(node)) for node in all_nodes}
    node_degrees = {node: len(neighbors) for node, neighbors in neighbor_sets.items()}
    
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
            
            # Combine metrics with weights
            similarities[i] = (0.3 * jaccard + 
                             0.2 * common_ratio + 
                             0.2 * resource_alloc + 
                             0.15 * pref_attach + 
                             0.15 * adamic_adar)
            
        except:
            continue
    
    return similarities

def predict_community_membership(G, node_pairs, best_partition, similarities):
    predictions = np.zeros(len(node_pairs), dtype=np.int32)
    
    for i, (node1, node2) in enumerate(node_pairs):
        # Check if both nodes exist in the partition
        if node1 in best_partition and node2 in best_partition:
            # Base prediction
            predictions[i] = 1 if best_partition[node1] == best_partition[node2] else 0
            
            # If not in same community, check additional features
            if predictions[i] == 0:
                try:
                    # Check path length
                    path_length = nx.shortest_path_length(G, node1, node2)
                    if path_length <= 2 and similarities[i] > 0.25:
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
                        
                        # Check if nodes have high similarity
                        if similarities[i] > 0.35:
                            predictions[i] = 1
                except:
                    pass
        else:
            # Fallback for nodes not in partition
            if similarities[i] > 0.35:
                predictions[i] = 1
    
    return predictions

def main():
    # File paths
    train_file = r"c:\Users\USER\Documents\NTUST\2nd Sem\Social Network\HW2_Community_Detection\HW2_Community_Detection\train.csv"
    test_file = r"c:\Users\USER\Documents\NTUST\2nd Sem\Social Network\HW2_Community_Detection\HW2_Community_Detection\test.csv"
    output_file = r"c:\Users\USER\Documents\NTUST\2nd Sem\Social Network\HW2_Community_Detection\HW2_Community_Detection\sample_submission.csv"
    
    print("Loading data...")
    train_df, test_df = load_data(train_file, test_file)
    
    print("Creating graph...")
    G = create_graph(train_df)
    
    print("Finding best community partition...")
    resolutions = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
    best_overall_partition = None
    best_overall_modularity = -float('inf')
    
    for res in resolutions:
        partition, modularity_score = get_best_partition(G, resolution=res, n_runs=20)
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