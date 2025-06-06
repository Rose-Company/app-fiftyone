import fiftyone as fo
from fiftyone import ViewField as F

# --- Configuration ---
# !!! PLEASE UPDATE THESE VALUES AS NEEDED !!!
DATASET_NAME = "video_dataset_faces_dlib_test"  # Dataset containing the samples with face detections
TARGET_CLUSTER_ID = 1                 # The specific cluster ID you want to filter for
DETECTION_FIELD = "dlib_face_detections_field" # The field containing fo.Detections
CLUSTER_ATTRIBUTE_NAME = "my_hdbscan_face_cluster" # Attribute on each detection
# --- --------------- ---

def filter_samples_by_cluster_id(
    dataset_name: str,
    detection_field: str,
    cluster_attribute_name: str,
    target_cluster_id: int
):
    """
    Filters samples in a FiftyOne dataset based on a cluster ID
    attribute within a Detections field.
    """
    # Connect to MongoDB (update if your URI is different)
    fo.config.database_uri = "mongodb://mongo:27017"

    if not fo.dataset_exists(dataset_name):
        print(f"Dataset '{dataset_name}' not found. Please check the DATASET_NAME.")
        return

    dataset = fo.load_dataset(dataset_name)
    print(f"Successfully loaded dataset '{dataset_name}' with {len(dataset)} total samples.")

    # Construct the query:
    # We are looking for samples where the list of cluster IDs from all detections
    # in 'detection_field' contains the 'target_cluster_id'.
    # F(detection_field + ".detections") gets the array of detection objects.
    # .map(F(cluster_attribute_name)) extracts the cluster ID from each detection.
    # .contains(target_cluster_id) checks if the target ID is in the list of extracted IDs.
    
    print(f"Filtering for samples where any detection in '{detection_field}' has '{cluster_attribute_name}' == {target_cluster_id}...")
    
    filtered_view = dataset.match(
        F(detection_field + ".detections")
        .map(F(cluster_attribute_name))
        .contains(target_cluster_id)
    )

    num_matching_samples = len(filtered_view)
    print(f"\\nFound {num_matching_samples} samples containing at least one detection with cluster ID '{target_cluster_id}'.")

    if num_matching_samples > 0:
        print("\\nFirst few matching samples (up to 5):")
        for i, sample in enumerate(filtered_view.limit(5)):
            print(f"  {i+1}. Sample ID: {sample.id}, Filepath: {sample.filepath}")
            
            # Optionally, show which detections matched
            matching_detection_details = []
            if sample[detection_field] and sample[detection_field].detections:
                for det_idx, det in enumerate(sample[detection_field].detections):
                    if det[cluster_attribute_name] == target_cluster_id:
                        detail = f"Detection {det_idx} (label: {det.label if det.label else 'N/A'})"
                        matching_detection_details.append(detail)
            if matching_detection_details:
                print(f"    Matching detections in this sample: {', '.join(matching_detection_details)}")
            else:
                print(f"    No detections seemed to match (this shouldn't happen with this query, check logic if it does).")

        print(f"\\n--- Launching FiftyOne App with the filtered view ---")
        print(f"View will contain {num_matching_samples} samples.")
        print(f"Dataset: {dataset_name}")
        print(f"Target Cluster ID: {target_cluster_id}")
        
        session = fo.launch_app(view=filtered_view)
        print("\\nFiftyOne App launched. Press Ctrl+C in this terminal to close the session when done.")
        session.wait()

    else: # No matching samples found
        print(f"\\nNo samples found for cluster ID {target_cluster_id}. Nothing to display in the App.")


    print(f"\\n--- Script Finished ---")


if __name__ == "__main__":
    print("--- Starting script to filter samples by HDBScan cluster ID ---")
    
    # Ensure the configuration at the top of the script is correct for your setup
    print(f"Using configuration: ")
    print(f"  DATASET_NAME: {DATASET_NAME}")
    print(f"  TARGET_CLUSTER_ID: {TARGET_CLUSTER_ID}")
    print(f"  DETECTION_FIELD: {DETECTION_FIELD}")
    print(f"  CLUSTER_ATTRIBUTE_NAME: {CLUSTER_ATTRIBUTE_NAME}")
    print("\\n")
    
    filter_samples_by_cluster_id(
        dataset_name=DATASET_NAME,
        detection_field=DETECTION_FIELD,
        cluster_attribute_name=CLUSTER_ATTRIBUTE_NAME,
        target_cluster_id=TARGET_CLUSTER_ID
    ) 