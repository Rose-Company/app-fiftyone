import numpy as np
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F
from sklearn.metrics import pairwise
from infomap import Infomap  # Import Infomap
import matplotlib.colors as mcolors
import os
import gc
import time

# --- CONFIGURATION ---
# The dataset containing face embeddings to be clustered.
FACE_DATASET_NAME = "video_dataset_faces_deepface_arcface_retinaface_image_17/6"

# --- INFOMAP CLUSTERING PARAMETERS ---
# Following the methodology from: https://github.com/xiaoxiong74/face-cluster-by-infomap
# Number of nearest neighbors to consider for each face when building the graph.
# For small videos with 3-4 characters, a smaller K creates tighter, more precise clusters.
K_NEIGHBORS = 10

# Minimum cosine similarity for two faces to be connected by an edge.
# For small character sets, we can afford to be more strict to avoid false connections.
SIMILARITY_THRESHOLD = 0.4

# Any cluster (community) with fewer members than this will be labeled 'unknown'.
# For small videos, keep this low to avoid losing small but valid character appearances.
MIN_COMMUNITY_SIZE = 3

# Whether to use FAISS for faster KNN search (recommended for large datasets)
USE_FAISS = False
# --- END CONFIGURATION ---

def build_knn_graph_faiss(embeddings, k, threshold):
    """
    Builds KNN graph using FAISS for faster computation, following the original methodology.
    This is the key optimization mentioned in the original paper for handling large datasets.
    
    Args:
        embeddings (np.ndarray): Array of face embeddings.
        k (int): Number of nearest neighbors to find for each node.
        threshold (float): Minimum similarity to create an edge.
        
    Returns:
        tuple: (edge_list, num_edges) where edge_list contains (i, j, weight) tuples
    """
    try:
        import faiss
        print(f"Using FAISS for faster KNN graph construction (k={k}, threshold={threshold})...")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index.add(embeddings_normalized.astype('float32'))
        
        # Search for k+1 nearest neighbors (including self)
        similarities, indices = index.search(embeddings_normalized.astype('float32'), k + 1)
        
        # Build edge list
        edges = []
        num_nodes = len(embeddings)
        
        for i in range(num_nodes):
            for j_idx in range(1, k + 1):  # Skip self (index 0)
                j = indices[i, j_idx]
                similarity = similarities[i, j_idx]
                
                if similarity > threshold and i != j:
                    edges.append((i, j, float(similarity)))
        
        print(f"FAISS-based graph built with {len(edges)} edges from {num_nodes} nodes.")
        return edges, len(edges)
        
    except ImportError:
        print("FAISS not available, falling back to sklearn-based method...")
        return build_knn_graph_sklearn(embeddings, k, threshold)

def build_knn_graph_sklearn(embeddings, k, threshold):
    """
    Fallback method using sklearn when FAISS is not available.
    """
    print(f"Using sklearn for KNN graph construction (k={k}, threshold={threshold})...")
    
    # Calculate pairwise cosine similarity
    similarity_matrix = pairwise.cosine_similarity(embeddings)
    num_nodes = embeddings.shape[0]
    
    edges = []
    for i in range(num_nodes):
        # Find k+1 nearest neighbors because the node itself is always the nearest
        neighbor_indices = np.argsort(similarity_matrix[i, :])[-(k+1):]
        
        for j in neighbor_indices:
            if i == j:
                continue
            
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                edges.append((i, j, float(similarity)))
                
    print(f"Sklearn-based graph built with {len(edges)} edges from {num_nodes} nodes.")
    return edges, len(edges)

def run_infomap_clustering(embeddings, k, threshold):
    """
    Builds a graph from embeddings and runs Infomap to find communities.
    Following the methodology from: https://github.com/xiaoxiong74/face-cluster-by-infomap

    Args:
        embeddings (np.ndarray): Array of face embeddings.
        k (int): Number of nearest neighbors to find for each node.
        threshold (float): Minimum similarity to create an edge.

    Returns:
        dict: A mapping from node index to community id.
    """
    print(f"\nRunning Infomap clustering with k={k} and threshold={threshold}...")
    print("Method: Learning to Cluster Faces by Infomap (xiaoxiong74/face-cluster-by-infomap)")
    
    # Step 1: Build KNN graph with optional FAISS acceleration
    if USE_FAISS:
        edges, edge_count = build_knn_graph_faiss(embeddings, k, threshold)
    else:
        edges, edge_count = build_knn_graph_sklearn(embeddings, k, threshold)
    
    # Step 2: Initialize Infomap
    # Using two-level algorithm as recommended for community detection
    im = Infomap("--two-level --silent")
    
    # Step 3: Add edges to Infomap
    print("Adding edges to Infomap...")
    for i, j, weight in edges:
        im.add_link(i, j, weight=weight)
                
    # Step 4: Run the Infomap algorithm
    print("Running Infomap algorithm...")
    im.run()
    
    print(f"Infomap found {im.num_top_modules} communities.")
    
    # Step 5: Return the cluster assignments for each node
    return im.get_modules()

def get_labels_from_infomap_modules(modules, num_nodes, min_community_size):
    """
    Converts Infomap modules into a labels array.
    Small communities are marked as noise (-1).
    """
    print(f"Processing {len(modules)} found communities (min size: {min_community_size})...")
    
    # Create a mapping from community_id to a list of its member nodes
    community_map = {}
    for node_index, community_id in modules.items():
        if community_id not in community_map:
            community_map[community_id] = []
        community_map[community_id].append(node_index)

    labels = np.full(num_nodes, -1, dtype=int)
    character_id_counter = 0
    
    valid_communities = 0
    # Sort communities by size to get consistent numbering (optional but good practice)
    sorted_communities = sorted(community_map.values(), key=len, reverse=True)

    for community_nodes in sorted_communities:
        if len(community_nodes) >= min_community_size:
            valid_communities += 1
            for node_index in community_nodes:
                labels[node_index] = character_id_counter
            character_id_counter += 1
            
    print(f"Found {valid_communities} valid character clusters.")
    return labels

def group_faces_into_characters_infomap():
    """
    Main function to cluster faces using Infomap community detection.
    """
    print("="*70)
    print("====== Face Clustering with Infomap Community Detection ======")
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
    print(f"\nFound distinct video groups to process: {distinct_video_ids}")

    for video_id in distinct_video_ids:
        group_name_for_logs = video_id if video_id is not None else "UNKNOWN_VIDEO"
        
        print(f"\n{'='*70}")
        print(f"Processing Group: {group_name_for_logs}")
        print(f"{'='*70}")

        video_view = face_dataset.match(F("video_id") == video_id)
        
        if not video_view:
            print("View is empty. Skipping.")
            continue
        
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
            print("No embeddings found for this group. Skipping.")
            continue
        
        embeddings_array = np.array(all_embeddings)
        print(f"Total embeddings for this group: {len(embeddings_array)}")
        
        # --- Step 2: Run Infomap Clustering ---
        modules = run_infomap_clustering(embeddings_array, k=K_NEIGHBORS, threshold=SIMILARITY_THRESHOLD)
        
        # --- Step 3: Convert modules to labels and filter small ones ---
        labels = get_labels_from_infomap_modules(modules, len(embeddings_array), MIN_COMMUNITY_SIZE)
        
        # --- Step 4: Save Detections to FiftyOne ---
        print("\nSaving final character detections to dataset...")
        colors = list(mcolors.TABLEAU_COLORS.values())
        unknown_color = "#808080"  # Gray for 'unknown'
        
        label_prefix = video_id if video_id is not None else "unknown_video"
        
        field_name = "character_detections_infomap"
        
        # We will iterate through all samples in the view and overwrite the field.
        # This avoids errors with clearing a field that may not exist on all samples yet.

        for sample in video_view.iter_samples(autosave=True):
            if sample.id not in sample_map:
                # If there are no embeddings for this sample, ensure the detection field is cleared
                sample[field_name] = None
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
            
            sample[field_name] = fol.Detections(detections=detections)
        
        final_cluster_count = len(set(l for l in labels if l != -1))
        group_name = video_id if video_id is not None else "UNKNOWN_VIDEO"
        print(f"\nGroup '{group_name}' processing complete. Found {final_cluster_count} final characters.")
        print(f"Results saved to field: '{field_name}'")

    print("\n\nAll videos processed.")

if __name__ == "__main__":
    group_faces_into_characters_infomap() 