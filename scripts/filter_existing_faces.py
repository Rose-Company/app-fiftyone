import cv2
import numpy as np
import fiftyone as fo
import os
import gc
import time

# --- CONFIGURATION ---
# IMPORTANT: Set this to the name of your existing, UNFILTERED dataset.
# This is the dataset that contains all the faces you want to apply the filter to.
SOURCE_DATASET_NAME = "video_dataset_faces_deepface_arcface_retinaface" 

# Name for the new dataset that will store the filtered results.
DESTINATION_DATASET_NAME = f"{SOURCE_DATASET_NAME}_blur_filtered"

# Filtering parameters
# This is the key parameter to tune. 100.0 was too high. 
# Try a lower value like 50.0. If still too many faces are filtered,
# try 30.0 or 20.0.
BLUR_THRESHOLD = 50.0
# --- END CONFIGURATION ---

def is_blurry(image, threshold):
    """
    Check if an image is blurry using Laplacian Variance.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def filter_dataset_by_blur():
    """
    Loads an existing dataset with face embeddings, applies a blur filter
    to each face, and saves the high-quality results to a new dataset.
    """
    print("="*60)
    print("=== Post-Processing: Filter Existing Dataset by Blurriness ===")
    print(f"Source Dataset: {SOURCE_DATASET_NAME}")
    print(f"Destination Dataset: {DESTINATION_DATASET_NAME}")
    print(f"Blur Threshold: {BLUR_THRESHOLD}")
    print("="*60)

    # Connect to MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Load source dataset
    if not fo.dataset_exists(SOURCE_DATASET_NAME):
        print(f"FATAL ERROR: Source dataset '{SOURCE_DATASET_NAME}' not found.")
        print("Please check the SOURCE_DATASET_NAME in the script configuration.")
        return
    source_dataset = fo.load_dataset(SOURCE_DATASET_NAME)

    # Create destination dataset
    if fo.dataset_exists(DESTINATION_DATASET_NAME):
        print(f"Destination dataset '{DESTINATION_DATASET_NAME}' already exists. Deleting.")
        fo.delete_dataset(DESTINATION_DATASET_NAME)
    destination_dataset = fo.Dataset(DESTINATION_DATASET_NAME)
    destination_dataset.persistent = True
    print(f"Created new destination dataset: '{DESTINATION_DATASET_NAME}'")

    total_faces_processed = 0
    total_faces_kept = 0
    start_time = time.time()
    
    with fo.ProgressBar(total=len(source_dataset)) as pb:
        for sample in source_dataset.iter_samples(autosave=False, progress=pb):
            if not sample.face_embeddings:
                continue

            try:
                frame_img = cv2.imread(sample.filepath)
                if frame_img is None: continue
                h, w = frame_img.shape[:2]

                high_quality_embeddings = []
                for face_data in sample.face_embeddings:
                    total_faces_processed += 1
                    
                    # Re-crop the face from the original frame
                    bbox = face_data["bounding_box"]
                    x_abs = int(bbox[0] * w)
                    y_abs = int(bbox[1] * h)
                    w_abs = int(bbox[2] * w)
                    h_abs = int(bbox[3] * h)
                    
                    face_crop = frame_img[y_abs : y_abs + h_abs, x_abs : x_abs + w_abs]

                    if face_crop.size == 0:
                        continue

                    # Apply blur filter
                    if not is_blurry(face_crop, BLUR_THRESHOLD):
                        high_quality_embeddings.append(face_data)
                        total_faces_kept += 1
                
                # If any faces passed the filter, create a new sample
                if high_quality_embeddings:
                    new_sample = sample.copy()
                    new_sample["face_embeddings"] = high_quality_embeddings
                    destination_dataset.add_sample(new_sample)

            except Exception as e:
                print(f"Error processing sample {sample.id}: {e}")
                continue

    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "="*60)
    print("=== FILTERING COMPLETE ===")
    print(f"Total time: {total_time:.2f} seconds.")
    print("-" * 30)
    print(f"Total faces processed: {total_faces_processed}")
    print(f"Total faces kept (not blurry): {total_faces_kept}")
    if total_faces_processed > 0:
        kept_ratio = total_faces_kept / total_faces_processed * 100
        print(f"Tỉ lệ giữ lại: {kept_ratio:.2f}%")
    print("-" * 30)
    print(f"Filtered data saved to dataset: '{DESTINATION_DATASET_NAME}'")
    print("="*60)


if __name__ == "__main__":
    filter_dataset_by_blur() 