import numpy as np
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F
from sklearn.metrics import pairwise
from sklearn.preprocessing import normalize
from infomap import Infomap
import networkx as nx
import matplotlib.colors as mcolors
import os
import gc
import time
from scipy.spatial.distance import cdist

# --- CONFIGURATION ---
# The dataset containing face embeddings to be clustered.
FACE_DATASET_NAME = "video_dataset_faces_deepface_arcface_retinaface_image_17/6"

# --- CONFIDENCE-GCN PARAMETERS ---
# Based on "Face Clustering via Graph Convolutional Networks with Confidence Edges" (ICCV 2023)
# https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Face_Clustering_via_Graph_Convolutional_Networks_with_Confidence_Edges_ICCV_2023_paper.pdf

# Target video to process (set to None to process all videos)
TARGET_VIDEO_ID = "test"

# === OPTIMIZED FOR 2-3 CHARACTER VIDEOS ===
# Initial kNN graph construction
K_INITIAL = 12              # Increased from 8 - more neighbors for better connectivity
SIMILARITY_THRESHOLD = 0.6  # Decreased from 0.65 - slightly more lenient to reduce noise

# Local Information Fusion (LIF) parameters
USE_LIF = True              # Enable Local Information Fusion
LIF_ALPHA = 0.85           # Decreased from 0.9 - balance between original and fused similarity
LIF_BETA = 0.15            # Increased from 0.1 - more weight on common neighbors
LIF_GAMMA = 0.85           # Decreased from 0.9 - balance similarity consistency

# Unsupervised Neighbor Determination (UND) parameters
USE_UND = True              # Enable Unsupervised Neighbor Determination
UND_THRESHOLD = 0.15       # Decreased from 0.2 - less strict filtering
MIN_NEIGHBORS = 4          # Increased from 3 - require more neighbors for stability
MAX_NEIGHBORS = 15         # Increased from 10 - allow more connections

# Confidence edge parameters
CONFIDENCE_THRESHOLD = 0.6  # Decreased from 0.65 - slightly more lenient to reduce noise
MIN_COMMUNITY_SIZE = 5      # Increased from 3 - require larger communities

# Output field name
OUTPUT_FIELD = "character_detections_confidence_gcn"
# --- END CONFIGURATION ---

def calculate_cosine_similarity_matrix(embeddings):
    """Calculate cosine similarity matrix for embeddings."""
    normalized_embeddings = normalize(embeddings, norm='l2')
    return np.dot(normalized_embeddings, normalized_embeddings.T)

def build_initial_knn_graph(embeddings, k):
    """Build initial kNN graph based on cosine similarity."""
    print(f"Building initial kNN graph with k={k}...")
    
    similarity_matrix = calculate_cosine_similarity_matrix(embeddings)
    n_nodes = len(embeddings)
    
    # Build adjacency list representation
    adjacency = {}
    for i in range(n_nodes):
        # Get k+1 nearest neighbors (including self)
        neighbor_indices = np.argsort(similarity_matrix[i])[::-1][:k+1]
        # Remove self
        neighbors = [(j, similarity_matrix[i, j]) for j in neighbor_indices if j != i]
        adjacency[i] = neighbors[:k]  # Keep only k neighbors
    
    print(f"Initial kNN graph built with {n_nodes} nodes.")
    return adjacency, similarity_matrix

def local_information_fusion(adjacency, similarity_matrix, alpha=0.7, beta=0.3, gamma=0.7):
    """
    Local Information Fusion (LIF) to improve similarity metric.
    Based on Section 3.2 of the paper.
    """
    if not USE_LIF:
        return similarity_matrix
    
    print("Applying Local Information Fusion (LIF)...")
    n_nodes = similarity_matrix.shape[0]
    lif_similarity = similarity_matrix.copy()
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # Get neighbors of i and j
            neighbors_i = set([neighbor[0] for neighbor in adjacency.get(i, [])])
            neighbors_j = set([neighbor[0] for neighbor in adjacency.get(j, [])])
            
            # Find common neighbors
            common_neighbors = neighbors_i.intersection(neighbors_j)
            
            if len(common_neighbors) > 0:
                # Calculate common neighbor count component
                cn_count = len(common_neighbors)
                max_possible_cn = min(len(neighbors_i), len(neighbors_j))
                cn_ratio = cn_count / max(max_possible_cn, 1)
                
                # Calculate similarity difference component
                sim_diffs = []
                for cn in common_neighbors:
                    sim_i_cn = similarity_matrix[i, cn]
                    sim_j_cn = similarity_matrix[j, cn]
                    sim_diffs.append(abs(sim_i_cn - sim_j_cn))
                
                avg_sim_diff = np.mean(sim_diffs) if sim_diffs else 0
                sim_consistency = 1 - avg_sim_diff  # Higher consistency = lower difference
                
                # Combine components using LIF formula
                lif_component = beta * cn_ratio + gamma * sim_consistency
                
                # Update similarity with LIF
                original_sim = similarity_matrix[i, j]
                lif_similarity[i, j] = alpha * original_sim + (1 - alpha) * lif_component
                lif_similarity[j, i] = lif_similarity[i, j]  # Symmetric
    
    print("LIF applied successfully.")
    return lif_similarity

def unsupervised_neighbor_determination(adjacency, lif_similarity, threshold=0.1):
    """
    Unsupervised Neighbor Determination (UND) to filter low-quality edges.
    Based on Section 3.3 of the paper.
    """
    if not USE_UND:
        return adjacency
    
    print("Applying Unsupervised Neighbor Determination (UND)...")
    refined_adjacency = {}
    
    for node in adjacency:
        neighbors = adjacency[node]
        if len(neighbors) <= MIN_NEIGHBORS:
            # Keep all neighbors if too few
            refined_adjacency[node] = neighbors
            continue
        
        # Sort neighbors by similarity
        sorted_neighbors = sorted(neighbors, key=lambda x: lif_similarity[node, x[0]], reverse=True)
        
        # Apply UND filtering based on similarity differences
        filtered_neighbors = []
        prev_similarity = float('inf')
        
        for neighbor_id, original_sim in sorted_neighbors:
            current_similarity = lif_similarity[node, neighbor_id]
            
            # Check similarity difference threshold
            if prev_similarity == float('inf') or (prev_similarity - current_similarity) <= threshold:
                filtered_neighbors.append((neighbor_id, current_similarity))
                prev_similarity = current_similarity
            else:
                # Stop adding neighbors if similarity drops too much
                break
        
        # Ensure minimum and maximum neighbor constraints
        if len(filtered_neighbors) < MIN_NEIGHBORS:
            filtered_neighbors = sorted_neighbors[:MIN_NEIGHBORS]
        elif len(filtered_neighbors) > MAX_NEIGHBORS:
            filtered_neighbors = filtered_neighbors[:MAX_NEIGHBORS]
        
        refined_adjacency[node] = filtered_neighbors
    
    print(f"UND applied. Average neighbors per node: {np.mean([len(neighbors) for neighbors in refined_adjacency.values()]):.1f}")
    return refined_adjacency

def identify_confidence_edges(adjacency, lif_similarity, threshold):
    """
    Identify confidence edges based on similarity threshold.
    Based on Section 3.1 of the paper.
    """
    print(f"Identifying confidence edges with threshold={threshold}...")
    
    confidence_edges = []
    total_edges = 0
    
    for node in adjacency:
        for neighbor_id, _ in adjacency[node]:
            if node < neighbor_id:  # Avoid duplicate edges
                similarity = lif_similarity[node, neighbor_id]
                total_edges += 1
                
                if similarity >= threshold:
                    confidence_edges.append((node, neighbor_id, similarity))
    
    confidence_ratio = len(confidence_edges) / max(total_edges, 1) * 100
    print(f"Found {len(confidence_edges)} confidence edges out of {total_edges} total edges ({confidence_ratio:.1f}%)")
    
    return confidence_edges

def run_infomap_clustering_with_confidence(confidence_edges, n_nodes):
    """
    Run Infomap clustering using confidence edges.
    Following the paper's approach of using Infomap as final clustering.
    """
    print("Running Infomap clustering with confidence edges...")
    
    # Initialize Infomap
    im = Infomap("--two-level --silent")
    
    # Add confidence edges to Infomap
    for i, j, weight in confidence_edges:
        im.add_link(i, j, weight=weight)
    
    # Run Infomap algorithm
    im.run()
    
    print(f"Infomap found {im.num_top_modules} communities using confidence edges.")
    
    # Get cluster assignments
    modules = im.get_modules()
    
    # Convert to labels array
    labels = np.full(n_nodes, -1, dtype=int)
    for node_id, community_id in modules.items():
        labels[node_id] = community_id
    
    return labels

def confidence_gcn_clustering(embeddings):
    """
    Main Confidence-GCN clustering pipeline.
    Implements the approach from ICCV 2023 paper.
    """
    print("\n=== Confidence-GCN Face Clustering ===")
    print("Based on: Face Clustering via Graph Convolutional Networks with Confidence Edges (ICCV 2023)")
    
    n_nodes = len(embeddings)
    print(f"Processing {n_nodes} face embeddings...")
    
    # Step 1: Build initial kNN graph
    adjacency, similarity_matrix = build_initial_knn_graph(embeddings, K_INITIAL)
    
    # Step 2: Apply Local Information Fusion (LIF)
    lif_similarity = local_information_fusion(adjacency, similarity_matrix, LIF_ALPHA, LIF_BETA, LIF_GAMMA)
    
    # Step 3: Apply Unsupervised Neighbor Determination (UND)
    refined_adjacency = unsupervised_neighbor_determination(adjacency, lif_similarity, UND_THRESHOLD)
    
    # Step 4: Identify confidence edges
    confidence_edges = identify_confidence_edges(refined_adjacency, lif_similarity, CONFIDENCE_THRESHOLD)
    
    # Step 5: Run Infomap clustering with confidence edges
    labels = run_infomap_clustering_with_confidence(confidence_edges, n_nodes)
    
    # Step 6: Filter small communities
    unique_labels, counts = np.unique(labels, return_counts=True)
    small_communities = unique_labels[counts < MIN_COMMUNITY_SIZE]
    
    if len(small_communities) > 0:
        print(f"Filtering {len(small_communities)} small communities (size < {MIN_COMMUNITY_SIZE})")
        for small_community in small_communities:
            labels[labels == small_community] = -1
    
    final_communities = len(set(l for l in labels if l != -1))
    noise_count = np.sum(labels == -1)
    
    print(f"Final result: {final_communities} communities, {noise_count} noise points")
    
    return labels

def group_faces_confidence_gcn():
    """
    Main function implementing Confidence-GCN face clustering.
    """
    print("="*70)
    print("=== Confidence-GCN Face Clustering ===")
    print("=== Based on ICCV 2023 Paper ===")
    print("="*70)
    
    # Connect to MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    if not fo.dataset_exists(FACE_DATASET_NAME):
        print(f"Error: Dataset '{FACE_DATASET_NAME}' not found.")
        return
    
    face_dataset = fo.load_dataset(FACE_DATASET_NAME)
    print(f"Loaded dataset '{FACE_DATASET_NAME}' with {len(face_dataset)} samples.")
    
    if "video_id" not in face_dataset.get_field_schema():
        print("\nFATAL ERROR: Dataset does not contain 'video_id' field.")
        return

    distinct_video_ids = face_dataset.distinct("video_id")
    print(f"Found video groups: {distinct_video_ids}")

    # Process target video or all videos
    if TARGET_VIDEO_ID and TARGET_VIDEO_ID in distinct_video_ids:
        video_ids_to_process = [TARGET_VIDEO_ID]
        print(f"Processing target video: {TARGET_VIDEO_ID}")
    else:
        video_ids_to_process = distinct_video_ids
        if TARGET_VIDEO_ID:
            print(f"Target video '{TARGET_VIDEO_ID}' not found, processing all videos.")

    for video_id in video_ids_to_process:
        group_name = video_id if video_id is not None else "UNKNOWN_VIDEO"
        
        print(f"\n{'='*50}")
        print(f"Processing: {group_name}")
        print(f"{'='*50}")

        video_view = face_dataset.match(F("video_id") == video_id)
        
        if not video_view:
            print("View is empty. Skipping.")
            continue
        
        # Gather embeddings
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
            print("No embeddings found. Skipping.")
            continue
        
        embeddings_array = np.array(all_embeddings)
        print(f"Total embeddings: {len(embeddings_array)}")
        
        # Run Confidence-GCN clustering
        labels = confidence_gcn_clustering(embeddings_array)
        
        # Save results
        print("\n--- Saving results ---")
        colors = list(mcolors.TABLEAU_COLORS.values())
        unknown_color = "#808080"
        
        label_prefix = video_id if video_id is not None else "unknown_video"
        
        for sample in video_view.iter_samples(autosave=True):
            if sample.id not in sample_map:
                sample[OUTPUT_FIELD] = None
                continue

            sample_info = sample_map[sample.id]
            sample_labels = labels[sample_info["slice"]]
            sample_bboxes = sample_info["bboxes"]
            
            detections = []
            for i, label in enumerate(sample_labels):
                character_id = int(label)
                
                if character_id == -1:
                    detection_label = f"{label_prefix}_unknown"
                    detection_color = unknown_color
                else:
                    detection_label = f"{label_prefix}_character_{character_id}"
                    detection_color = colors[character_id % len(colors)]

                detection = fol.Detection(
                    label=detection_label,
                    bounding_box=sample_bboxes[i],
                    character_id=character_id,
                    confidence=1.0, 
                    color=detection_color
                )
                detections.append(detection)
            
            sample[OUTPUT_FIELD] = fol.Detections(detections=detections)
        
        # Final summary
        final_communities = len(set(l for l in labels if l != -1))
        noise_count = np.sum(labels == -1)
        
        print(f"\nResults for '{group_name}':")
        print(f"  - Characters found: {final_communities}")
        print(f"  - Unknown faces: {noise_count}")
        print(f"  - Results saved to: '{OUTPUT_FIELD}'")

    print(f"\n\nProcessing complete! Check field '{OUTPUT_FIELD}' in FiftyOne.")

if __name__ == "__main__":
    group_faces_confidence_gcn() 