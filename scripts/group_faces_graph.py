import numpy as np
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F
from sklearn.metrics import pairwise
import networkx as nx
from networkx.algorithms import community as nx_comm
import matplotlib.colors as mcolors
import os
import gc
import time

# --- CONFIGURATION ---
# The dataset containing face embeddings to be clustered.
FACE_DATASET_NAME = "video_dataset_faces_deepface_arcface_retinaface_final"

# Video IDs to process - specify which videos to test
# Examples:
VIDEO_IDS_TO_PROCESS = ["test2"]  # Process only these specific videos
#VIDEO_IDS_TO_PROCESS = ["test1"]            # Process only one video
# VIDEO_IDS_TO_PROCESS = None                  # Process all videos (default behavior)
#VIDEO_IDS_TO_PROCESS = None  # Change this to specify which videos to process

# --- GRAPH CLUSTERING PARAMETERS ---
# Number of nearest neighbors to consider for each face when building the graph.
# A smaller value makes the graph sparser and clusters tighter.
K_NEIGHBORS = 50

# Minimum cosine similarity for two faces to be connected by an edge.
# This acts as a hard filter after finding the K-nearest neighbors.
SIMILARITY_THRESHOLD = 0.46
# Any cluster (community) with fewer members than this will be labeled 'unknown'.
MIN_COMMUNITY_SIZE = 3
# --- END CONFIGURATION ---

def build_affinity_graph(embeddings, k, threshold):
    """
    Builds a graph where nodes are faces and edges connect similar faces.

    Args:
        embeddings (np.ndarray): Array of face embeddings.
        k (int): Number of nearest neighbors to find for each node.
        threshold (float): Minimum similarity to create an edge.

    Returns:
        networkx.Graph: The constructed graph.
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
    Small communities are marked as noise (-1).
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

def group_faces_into_characters_graph():
    """
    Main function to cluster faces using a graph-based community detection approach.
    """
    print("="*70)
    print("=== Advanced Face Clustering with Graph Community Detection ===")
    print("="*70)

    # Connect to MongoDB
    database_uri = os.getenv("FIFTYONE_DATABASE_URI")
    if not database_uri:
        print("Error: FIFTYONE_DATABASE_URI environment variable is required!")
        exit(1)
    fo.config.database_uri = database_uri
        
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

    # Filter video_ids if specified
    if VIDEO_IDS_TO_PROCESS is not None:
        print(f"\nLọc video_ids theo cấu hình: {VIDEO_IDS_TO_PROCESS}")
        
        # Validate requested video_ids
        available_video_ids = set(distinct_video_ids)
        requested_ids = set(VIDEO_IDS_TO_PROCESS)
        valid_ids = requested_ids.intersection(available_video_ids)
        invalid_ids = requested_ids - available_video_ids
        
        if invalid_ids:
            print(f"CẢNH BÁO: Các video_id không tồn tại: {sorted(invalid_ids)}")
        
        if not valid_ids:
            print("KHÔNG CÓ VIDEO_ID HỢP LỆ ĐỂ XỬ LÝ!")
            return
        
        print(f"Sẽ xử lý các video_id: {sorted(valid_ids)}")
        distinct_video_ids = list(valid_ids)
    else:
        print("Xử lý tất cả video_ids")

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
        
        # --- Step 2: Build Affinity Graph ---
        graph = build_affinity_graph(embeddings_array, k=K_NEIGHBORS, threshold=SIMILARITY_THRESHOLD)

        # --- Step 3: Find Communities (Clusters) ---
        print("\nFinding communities using Louvain method...")
        # Note: Louvain is good for performance. For potentially higher quality on some graphs,
        # you could try other algorithms like Girvan-Newman, but they are much slower.
        communities = nx_comm.louvain_communities(graph, weight='weight', seed=42)
        
        # --- Step 4: Convert communities to labels and filter small ones ---
        labels = get_labels_from_communities(communities, len(embeddings_array), MIN_COMMUNITY_SIZE)
        
        # --- Step 5: Save Detections to FiftyOne ---
        print("\nSaving final character detections to dataset...")
        colors = list(mcolors.TABLEAU_COLORS.values())
        unknown_color = "#808080"  # Gray for 'unknown'
        
        label_prefix = video_id if video_id is not None else "unknown_video"
        
        for sample in video_view.iter_samples(autosave=True):
            if sample.id not in sample_map:
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
            
            sample["character_detections"] = fol.Detections(detections=detections)
        
        final_cluster_count = len(set(l for l in labels if l != -1))
        group_name = video_id if video_id is not None else "UNKNOWN_VIDEO"
        print(f"\nGroup '{group_name}' processing complete. Found {final_cluster_count} final characters.")

    # Final summary
    print("\n" + "="*70)
    print("=== KẾT QUẢ CLUSTERING HOÀN TẤT ===")
    if VIDEO_IDS_TO_PROCESS is not None:
        print(f"Video IDs đã xử lý: {VIDEO_IDS_TO_PROCESS}")
        print(f"Số lượng videos đã clustering: {len(distinct_video_ids)}")
    else:
        print(f"Đã clustering tất cả {len(distinct_video_ids)} videos")
    print("="*70)

    print("\n\nAll videos processed. Check the 'character_detections' field in FiftyOne.")

if __name__ == "__main__":
    group_faces_into_characters_graph() 