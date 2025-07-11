import numpy as np
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F
from sklearn.metrics import pairwise
import networkx as nx
from networkx.algorithms import community as nx_comm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import gc
import time
from datetime import datetime

# --- CONFIGURATION ---
# The dataset containing face embeddings to be clustered.
FACE_DATASET_NAME = "video_dataset_faces_deepface_arcface_retinaface"

# --- GRAPH CLUSTERING PARAMETERS ---
# Copy parameters from group_faces_graph.py
K_NEIGHBORS = 53
SIMILARITY_THRESHOLD = 0.5
MIN_COMMUNITY_SIZE = 3

# --- VISUALIZATION PARAMETERS ---
# Which video to visualize (set to None to visualize the first video found)
TARGET_VIDEO_ID = "test"  # Change this to the video you want to visualize

# Output directory for PNG files
OUTPUT_DIR = "/app/data/graph_visualizations"

# Graph layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
LAYOUT_ALGORITHM = "spring"

# Figure size (width, height) in inches
FIGURE_SIZE = (15, 12)
# --- END CONFIGURATION ---

def build_affinity_graph(embeddings, k, threshold):
    """
    Builds a graph where nodes are faces and edges connect similar faces.
    (Copied from group_faces_graph.py)
    """
    print(f"\nBuilding graph with k={k} and threshold={threshold}...")
    
    # 1. Calculate pairwise cosine similarity
    similarity_matrix = pairwise.cosine_similarity(embeddings)
    
    # 2. Build the graph
    graph = nx.Graph()
    num_nodes = embeddings.shape[0]
    graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        # Find k+1 nearest neighbors because the node itself is always the nearest
        neighbor_indices = np.argsort(similarity_matrix[i, :])[-(k+1):]
        
        for j in neighbor_indices:
            # Don't connect a node to itself
            if i == j:
                continue
            
            # Add edge if similarity is above the threshold
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                graph.add_edge(i, j, weight=similarity)
                
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

def get_labels_from_communities(communities, num_nodes, min_community_size):
    """
    Converts a list of communities into a labels array.
    (Copied from group_faces_graph.py)
    """
    print(f"Processing {len(communities)} found communities (min size: {min_community_size})...")
    labels = np.full(num_nodes, -1, dtype=int)
    character_id = 0
    
    valid_communities = 0
    for community in communities:
        if len(community) >= min_community_size:
            valid_communities += 1
            for node_index in community:
                labels[node_index] = character_id
            character_id += 1
            
    print(f"Found {valid_communities} valid character clusters.")
    return labels

def visualize_face_graph():
    """
    Main function to visualize the face clustering graph and save as PNG.
    """
    print("="*70)
    print("=== Face Clustering Graph Visualization ===")
    print("="*70)

    # Connect to MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    if not fo.dataset_exists(FACE_DATASET_NAME):
        print(f"Error: Dataset '{FACE_DATASET_NAME}' not found.")
        return
    
    face_dataset = fo.load_dataset(FACE_DATASET_NAME)
    print(f"Loaded dataset '{FACE_DATASET_NAME}' with {len(face_dataset)} samples (frames).")
    
    if "video_id" not in face_dataset.get_field_schema():
        print("\nFATAL ERROR: Dataset does not contain 'video_id' field.")
        print("Please run 'scripts/add_video_id_to_dataset.py' first.")
        return

    distinct_video_ids = face_dataset.distinct("video_id")
    print(f"\nFound distinct video groups: {distinct_video_ids}")

    # Select target video
    if TARGET_VIDEO_ID and TARGET_VIDEO_ID in distinct_video_ids:
        video_id = TARGET_VIDEO_ID
    else:
        video_id = distinct_video_ids[0]
        print(f"Target video '{TARGET_VIDEO_ID}' not found, using '{video_id}' instead.")
    
    print(f"\nVisualizing video: {video_id}")
    
    video_view = face_dataset.match(F("video_id") == video_id)
    
    if not video_view:
        print("View is empty. Exiting.")
        return
    
    # --- Step 1: Gather Embeddings ---
    all_embeddings, sample_map = [], {}
    print("Gathering embeddings...")
    for sample in video_view:
        if sample.face_embeddings:
            start_index = len(all_embeddings)
            embeddings_in_sample = [fe["embedding"] for fe in sample.face_embeddings]
            all_embeddings.extend(embeddings_in_sample)
            end_index = len(all_embeddings)
            sample_map[sample.id] = {
                "slice": slice(start_index, end_index),
                "bboxes": [fe["bounding_box"] for fe in sample.face_embeddings]
            }
    
    if not all_embeddings:
        print("No embeddings found for this video. Exiting.")
        return
    
    embeddings_array = np.array(all_embeddings)
    print(f"Total embeddings for this video: {len(embeddings_array)}")
    
    # --- Step 2: Build Affinity Graph ---
    graph = build_affinity_graph(embeddings_array, k=K_NEIGHBORS, threshold=SIMILARITY_THRESHOLD)

    # --- Step 3: Find Communities (Clusters) ---
    print("\nFinding communities using Louvain method...")
    communities = nx_comm.louvain_communities(graph, weight='weight', seed=42)
    
    # --- Step 4: Convert communities to labels ---
    labels = get_labels_from_communities(communities, len(embeddings_array), MIN_COMMUNITY_SIZE)
    
    # --- Step 5: Visualize the Graph ---
    print(f"\nCreating visualization...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=FIGURE_SIZE)
    plt.title(f'Face Clustering Graph - Video: {video_id}\n'
              f'Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}, '
              f'Clusters: {len(set(l for l in labels if l != -1))}', 
              fontsize=16, fontweight='bold')
    
    # Choose layout algorithm
    if LAYOUT_ALGORITHM == "spring":
        pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
    elif LAYOUT_ALGORITHM == "circular":
        pos = nx.circular_layout(graph)
    elif LAYOUT_ALGORITHM == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif LAYOUT_ALGORITHM == "spectral":
        pos = nx.spectral_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=42)
    
    # Prepare colors for clusters
    colors = list(mcolors.TABLEAU_COLORS.values())
    unknown_color = "#808080"  # Gray for unknown/noise
    
    node_colors = []
    for i in range(len(labels)):
        if labels[i] == -1:
            node_colors.append(unknown_color)
        else:
            node_colors.append(colors[labels[i] % len(colors)])
    
    # Draw the graph
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, 
                          node_color=node_colors, 
                          node_size=100, 
                          alpha=0.8)
    
    # Add labels for cluster centers (optional)
    cluster_centers = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label not in cluster_centers:
                cluster_centers[label] = []
            cluster_centers[label].append(i)
    
    # Draw cluster labels
    for cluster_id, node_indices in cluster_centers.items():
        if len(node_indices) >= MIN_COMMUNITY_SIZE:
            # Find the center position of this cluster
            cluster_pos = np.mean([pos[i] for i in node_indices], axis=0)
            plt.text(cluster_pos[0], cluster_pos[1], f'C{cluster_id}', 
                    fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                    ha='center', va='center')
    
    # Create legend
    legend_elements = []
    for cluster_id in sorted(set(l for l in labels if l != -1)):
        color = colors[cluster_id % len(colors)]
        count = np.sum(labels == cluster_id)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10,
                                        label=f'Character {cluster_id} ({count} faces)'))
    
    # Add unknown/noise to legend if present
    unknown_count = np.sum(labels == -1)
    if unknown_count > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=unknown_color, markersize=10,
                                        label=f'Unknown ({unknown_count} faces)'))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Remove axes
    plt.axis('off')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"face_graph_{video_id}_{timestamp}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\nGraph visualization saved to: {filepath}")
    
    # Also save a summary text file
    summary_filename = f"face_graph_summary_{video_id}_{timestamp}.txt"
    summary_filepath = os.path.join(OUTPUT_DIR, summary_filename)
    
    with open(summary_filepath, 'w') as f:
        f.write(f"Face Clustering Graph Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Video ID: {video_id}\n")
        f.write(f"Total faces: {len(embeddings_array)}\n")
        f.write(f"Graph nodes: {graph.number_of_nodes()}\n")
        f.write(f"Graph edges: {graph.number_of_edges()}\n")
        f.write(f"Parameters: K={K_NEIGHBORS}, Threshold={SIMILARITY_THRESHOLD}, MinSize={MIN_COMMUNITY_SIZE}\n")
        f.write(f"Layout algorithm: {LAYOUT_ALGORITHM}\n\n")
        
        f.write("Cluster Distribution:\n")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label == -1:
                f.write(f"  Unknown: {count} faces\n")
            else:
                f.write(f"  Character {label}: {count} faces\n")
    
    print(f"Summary saved to: {summary_filepath}")
    
    # Show the plot (optional, comment out if running headless)
    # plt.show()
    
    plt.close()
    print("\nVisualization complete!")

if __name__ == "__main__":
    visualize_face_graph() 