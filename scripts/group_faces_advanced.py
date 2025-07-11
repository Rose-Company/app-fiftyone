import numpy as np
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors
import json
import os
import gc

# --- CONFIGURATION ---
# The dataset containing face embeddings to be clustered.
FACE_DATASET_NAME = "video_dataset_faces_deepface_arcface_retinaface"

# --- SIMPLE CLUSTERING PARAMETERS (Easy to adjust for testing) ---
# Which video to process (set to None to process all videos)
TARGET_VIDEO_ID = "test"  # Change this to test specific video

# HDBSCAN Parameters - Easy to adjust for testing
MIN_CLUSTER_SIZE = 5      # Minimum faces needed to form a character cluster
MIN_SAMPLES = 5           # Minimum samples in neighborhood for core point
CLUSTER_SELECTION_EPSILON = 0.0  # Distance threshold for cluster selection

# Alternative: Use simple clustering instead of HDBSCAN
USE_SIMPLE_CLUSTERING = False  # Set to True to use KMeans instead
N_CLUSTERS_SIMPLE = 4          # Number of clusters for simple clustering

# Noise handling
ASSIGN_NOISE_TO_LARGEST = True  # Assign noise points to nearest large cluster
MIN_CLUSTER_SIZE_FINAL = 3      # Final minimum size after noise assignment

# Output field name
OUTPUT_FIELD = "character_detections_simple"
# --- END CONFIGURATION ---

def cluster_faces_hdbscan_simple(embeddings_array, min_cluster_size, min_samples, cluster_selection_epsilon):
    """
    Simple HDBSCAN clustering with manual parameters.
    """
    try:
        import hdbscan
    except ImportError:
        print("HDBSCAN library not found. Please run: pip install hdbscan")
        print("Falling back to simple KMeans clustering...")
        return cluster_faces_simple(embeddings_array, N_CLUSTERS_SIMPLE)

    if embeddings_array.size == 0:
        return np.array([], dtype=int)

    # Normalize embeddings
    normalized_embeddings = normalize(embeddings_array)
    
    print(f"Running HDBSCAN with:")
    print(f"  - min_cluster_size: {min_cluster_size}")
    print(f"  - min_samples: {min_samples}")
    print(f"  - cluster_selection_epsilon: {cluster_selection_epsilon}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean',
        allow_single_cluster=True
    )
    
    labels = clusterer.fit_predict(normalized_embeddings)
    
    n_clusters = len(set(l for l in labels if l != -1))
    n_noise = np.sum(labels == -1)
    if len(labels) > 0:
        noise_ratio = n_noise / len(labels) * 100
        print(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points ({noise_ratio:.1f}%).")
    
    return labels

def cluster_faces_simple(embeddings_array, n_clusters):
    """
    Simple KMeans clustering for easy testing.
    """
    if embeddings_array.size == 0:
        return np.array([], dtype=int)
    
    if len(embeddings_array) < n_clusters:
        print(f"Warning: Only {len(embeddings_array)} faces, reducing clusters to {len(embeddings_array)}")
        n_clusters = len(embeddings_array)
    
    # Normalize embeddings
    normalized_embeddings = normalize(embeddings_array)
    
    print(f"Running KMeans with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized_embeddings)
    
    print(f"KMeans found {n_clusters} clusters.")
    return labels

def assign_noise_to_clusters(labels, embeddings_array):
    """
    Assign noise points (-1) to the nearest large cluster.
    """
    if not ASSIGN_NOISE_TO_LARGEST:
        return labels
    
    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) == 0:
        return labels
    
    # Find clusters that meet minimum size requirement
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    valid_clusters = unique_labels[counts >= MIN_CLUSTER_SIZE_FINAL]
    
    if len(valid_clusters) == 0:
        print("No valid clusters found for noise assignment.")
        return labels
    
    print(f"Assigning {len(noise_indices)} noise points to nearest clusters...")
    
    # Calculate cluster centroids
    cluster_centroids = {}
    for cluster_id in valid_clusters:
        cluster_mask = labels == cluster_id
        cluster_centroids[cluster_id] = np.mean(embeddings_array[cluster_mask], axis=0)
    
    # Assign each noise point to nearest cluster
    new_labels = labels.copy()
    for noise_idx in noise_indices:
        noise_embedding = embeddings_array[noise_idx]
        
        # Find nearest cluster
        min_distance = float('inf')
        nearest_cluster = -1
        
        for cluster_id, centroid in cluster_centroids.items():
            distance = np.linalg.norm(noise_embedding - centroid)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = cluster_id
        
        if nearest_cluster != -1:
            new_labels[noise_idx] = nearest_cluster
    
    n_reassigned = np.sum(new_labels != labels)
    print(f"Reassigned {n_reassigned} noise points to existing clusters.")
    
    return new_labels

def group_faces_into_characters_simple():
    """
    Simplified main function for easy testing and parameter adjustment.
    """
    print("="*70)
    print("=== Simple Face Clustering (Easy Testing) ===")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Target Video: {TARGET_VIDEO_ID}")
    print(f"  - Use Simple Clustering: {USE_SIMPLE_CLUSTERING}")
    if USE_SIMPLE_CLUSTERING:
        print(f"  - Number of Clusters: {N_CLUSTERS_SIMPLE}")
    else:
        print(f"  - Min Cluster Size: {MIN_CLUSTER_SIZE}")
        print(f"  - Min Samples: {MIN_SAMPLES}")
        print(f"  - Cluster Selection Epsilon: {CLUSTER_SELECTION_EPSILON}")
    print(f"  - Assign Noise to Largest: {ASSIGN_NOISE_TO_LARGEST}")
    print(f"  - Final Min Cluster Size: {MIN_CLUSTER_SIZE_FINAL}")
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

    # Process specific video or all videos
    if TARGET_VIDEO_ID and TARGET_VIDEO_ID in distinct_video_ids:
        video_ids_to_process = [TARGET_VIDEO_ID]
        print(f"Processing only target video: {TARGET_VIDEO_ID}")
    else:
        video_ids_to_process = distinct_video_ids
        if TARGET_VIDEO_ID:
            print(f"Target video '{TARGET_VIDEO_ID}' not found, processing all videos.")

    for video_id in video_ids_to_process:
        group_name_for_logs = video_id if video_id is not None else "UNKNOWN_VIDEO"
        
        print(f"\n{'='*50}")
        print(f"Processing: {group_name_for_logs}")
        print(f"{'='*50}")

        video_view = face_dataset.match(F("video_id") == video_id)
        
        if not video_view:
            print("View is empty. Skipping.")
            continue
        
        # Step 1: Gather embeddings
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
        
        # Step 2: Run clustering
        print("\n--- Clustering ---")
        if USE_SIMPLE_CLUSTERING:
            labels = cluster_faces_simple(embeddings_array, N_CLUSTERS_SIMPLE)
        else:
            labels = cluster_faces_hdbscan_simple(
                embeddings_array, 
                MIN_CLUSTER_SIZE, 
                MIN_SAMPLES, 
                CLUSTER_SELECTION_EPSILON
            )
        
        # Step 3: Handle noise points
        if not USE_SIMPLE_CLUSTERING:
            labels = assign_noise_to_clusters(labels, embeddings_array)
        
        # Step 4: Filter small clusters
        print("\n--- Final filtering ---")
        unique_labels, counts = np.unique(labels, return_counts=True)
        small_clusters = unique_labels[counts < MIN_CLUSTER_SIZE_FINAL]
        
        if len(small_clusters) > 0:
            print(f"Marking {len(small_clusters)} small clusters as unknown...")
            for small_cluster in small_clusters:
                labels[labels == small_cluster] = -1
        
        # Step 5: Save results
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
        final_cluster_count = len(set(l for l in labels if l != -1))
        final_unknown_count = np.sum(labels == -1)
        
        print(f"\nResults for '{group_name_for_logs}':")
        print(f"  - Characters found: {final_cluster_count}")
        print(f"  - Unknown faces: {final_unknown_count}")
        print(f"  - Results saved to field: '{OUTPUT_FIELD}'")
        
        # Show cluster distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"  - Cluster distribution:")
        for label, count in zip(unique_labels, counts):
            if label == -1:
                print(f"    Unknown: {count} faces")
            else:
                print(f"    Character {label}: {count} faces")

    print(f"\n\nProcessing complete! Check field '{OUTPUT_FIELD}' in FiftyOne.")

if __name__ == "__main__":
    group_faces_into_characters_simple() 