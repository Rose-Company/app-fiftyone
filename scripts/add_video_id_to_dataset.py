import fiftyone as fo
import os

# --- CONFIGURATION ---
# The name of the dataset to which you want to add the 'video_id' field.
DATASET_TO_UPDATE = "video_dataset_faces_dic_cnn"

# The name of the dataset that contains the original source videos (e.g., .mp4 files).
# The script will look up video filepaths from this dataset.
SOURCE_VIDEO_DATASET = "video_dataset"
# --- END CONFIGURATION ---

def add_video_id_field_from_source():
    """
    Adds or updates a 'video_id' field by linking back to the source video dataset.

    This script works by:
    1. Loading the source video dataset to create a map from a video's sample ID
       to its filename-based video_id.
    2. Iterating through the target dataset (containing frames).
    3. For each frame, it uses the frame's `sample_id` (which refers to the original
       video sample) to look up the correct video_id from the map.
    4. It then sets the `video_id` field on the frame sample.
    """
    print("="*60)
    print("=== Script to Add 'video_id' from Source Video Dataset ===")
    print(f"Target dataset: {DATASET_TO_UPDATE}")
    print(f"Source video dataset: {SOURCE_VIDEO_DATASET}")
    print("="*60)

    # Connect to MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"

    # Load datasets
    if not fo.dataset_exists(DATASET_TO_UPDATE):
        print(f"Error: Target dataset '{DATASET_TO_UPDATE}' not found.")
        return
    if not fo.dataset_exists(SOURCE_VIDEO_DATASET):
        print(f"Error: Source video dataset '{SOURCE_VIDEO_DATASET}' not found.")
        return

    dataset_to_update = fo.load_dataset(DATASET_TO_UPDATE)
    source_video_dataset = fo.load_dataset(SOURCE_VIDEO_DATASET)

    # Step 1: Create a mapping from source video sample_id to a clean video_id
    print("Building a map from source video Sample ID to video_id...")
    video_id_map = {}
    for video_sample in source_video_dataset.select_fields(["id", "filepath"]):
        video_path = video_sample.filepath
        video_name = os.path.basename(video_path)
        video_id = os.path.splitext(video_name)[0]
        video_id_map[video_sample.id] = video_id
    
    if not video_id_map:
        print("Error: Could not build video_id map. The source dataset might be empty.")
        return
        
    print(f"Map built successfully with {len(video_id_map)} entries.")

    # Step 2: Iterate over the target dataset and update the 'video_id' field
    print(f"\nScanning {len(dataset_to_update)} samples to add/update 'video_id'...")

    updated_count = 0
    with fo.ProgressBar(total=len(dataset_to_update)) as pb:
        for frame_sample in dataset_to_update.iter_samples(autosave=True, progress=pb):
            try:
                # The 'sample_id' of a frame sample refers to the ID of the source video sample
                source_video_sample_id = frame_sample.sample_id
                
                if source_video_sample_id in video_id_map:
                    correct_video_id = video_id_map[source_video_sample_id]
                    
                    # Update the field only if it's new or different
                    if "video_id" not in frame_sample or frame_sample.video_id != correct_video_id:
                        frame_sample["video_id"] = correct_video_id
                        updated_count += 1
                else:
                    # Fallback if a frame's source video is not in the map
                    if "video_id" not in frame_sample or frame_sample.video_id != "unknown_video":
                        frame_sample["video_id"] = "unknown_video"
                        updated_count += 1
                        
            except Exception as e:
                print(f"Could not process sample {frame_sample.id}: {e}")
                continue

    print("\n" + "="*60)
    print("=== UPDATE COMPLETE ===")
    if updated_count > 0:
        print(f"Successfully added or updated 'video_id' for {updated_count} samples in '{DATASET_TO_UPDATE}'.")
    else:
        print("All samples already had up-to-date 'video_id' fields. No changes were made.")
    
    print("\nYou can now run the 'group_faces_advanced.py' script.")
    print("="*60)

if __name__ == "__main__":
    add_video_id_field_from_source() 