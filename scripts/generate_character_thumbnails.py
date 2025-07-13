import fiftyone as fo
import os
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from datetime import datetime
import shutil

# MongoDB connection configuration
database_uri = os.getenv("FIFTYONE_DATABASE_URI")
if not database_uri:
    print("Error: FIFTYONE_DATABASE_URI environment variable is required!")
    exit(1)
fo.config.database_uri = database_uri

def calculate_quality_score(detection, frame_shape):
    """
    Calculate quality score for a face detection
    
    Args:
        detection: FiftyOne detection object
        frame_shape: (height, width) of the frame
        
    Returns:
        float: Quality score (higher is better)
    """
    # Get bounding box
    bbox = detection.bounding_box
    x, y, w, h = bbox
    
    # Convert relative coordinates to absolute
    frame_h, frame_w = frame_shape
    abs_x = int(x * frame_w)
    abs_y = int(y * frame_h)
    abs_w = int(w * frame_w)
    abs_h = int(h * frame_h)
    
    # Factor 1: Confidence score (0-1)
    confidence = detection.confidence if hasattr(detection, 'confidence') else 0.5
    
    # Factor 2: Face size (larger is better, normalized by frame size)
    face_area = abs_w * abs_h
    frame_area = frame_w * frame_h
    size_score = min(face_area / frame_area * 100, 1.0)  # Cap at 1.0
    
    # Factor 3: Position score (center is better)
    center_x = abs_x + abs_w / 2
    center_y = abs_y + abs_h / 2
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2
    
    # Distance from center (normalized)
    distance = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
    max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)
    position_score = 1.0 - (distance / max_distance)
    
    # Factor 4: Aspect ratio score (closer to 1:1 is better for faces)
    aspect_ratio = abs_w / abs_h if abs_h > 0 else 0
    aspect_score = 1.0 - abs(aspect_ratio - 1.0)  # Penalty for non-square faces
    aspect_score = max(aspect_score, 0.1)  # Minimum score
    
    # Weighted combination
    quality_score = (
        confidence * 0.4 +          # 40% confidence
        size_score * 0.3 +          # 30% size
        position_score * 0.2 +      # 20% position
        aspect_score * 0.1          # 10% aspect ratio
    )
    
    return quality_score

def crop_face_with_padding(image, bbox, padding_ratio=0.3, target_size=(200, 200)):
    """
    Crop face from image with padding and resize to target size
    
    Args:
        image: PIL Image or numpy array
        bbox: (x, y, w, h) in relative coordinates
        padding_ratio: Extra padding around face
        target_size: (width, height) for output
        
    Returns:
        PIL Image: Cropped and resized face
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    img_w, img_h = image.size
    x, y, w, h = bbox
    
    # Convert to absolute coordinates
    abs_x = int(x * img_w)
    abs_y = int(y * img_h)
    abs_w = int(w * img_w)
    abs_h = int(h * img_h)
    
    # Add padding
    padding_w = int(abs_w * padding_ratio)
    padding_h = int(abs_h * padding_ratio)
    
    # Calculate crop coordinates with padding
    crop_x1 = max(0, abs_x - padding_w)
    crop_y1 = max(0, abs_y - padding_h)
    crop_x2 = min(img_w, abs_x + abs_w + padding_w)
    crop_y2 = min(img_h, abs_y + abs_h + padding_h)
    
    # Crop the face
    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    # Resize to target size
    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    return resized

def enhance_thumbnail(image):
    """
    Apply basic enhancements to thumbnail
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image: Enhanced image
    """
    # Slight contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(1.1)
    
    # Slight sharpness enhancement
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(1.1)
    
    return enhanced

def load_character_index():
    """
    Load character index from individual video JSON files
    """
    character_index = {}
    data_dir = "/fiftyone/data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found")
        return {}
    
    # Load all existing video JSON files
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and not filename.startswith('character_'):
            # This is a video file (e.g., test1.json, test2.json)
            video_name = filename.replace('.json', '.mp4')
            file_path = os.path.join(data_dir, filename)
            
            try:
                with open(file_path, "r") as f:
                    character_index[video_name] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    return character_index

def generate_character_thumbnails():
    """
    Generate thumbnails for all characters using Face Crop + Quality Scoring
    """
    print("Starting character thumbnail generation...")
    
    # Load face dataset
    face_dataset_name = "video_dataset_faces_deepface_arcface_retinaface_final"
    if not fo.dataset_exists(face_dataset_name):
        print(f"Face dataset '{face_dataset_name}' not found. Please extract faces first.")
        return False
    
    face_dataset = fo.load_dataset(face_dataset_name)
    
    # Load character index
    character_index = load_character_index()
    if not character_index:
        print("No character index found. Please run character_search.py first.")
        return False
    
    # Create thumbnails directory
    thumbnails_dir = "/fiftyone/data/thumbnails"
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    # Collect all characters from all videos and extract base character names
    all_characters = set()
    character_to_video_mapping = {}
    
    for video_name, characters in character_index.items():
        for character_name in characters.keys():
            # Extract base character name (remove video suffix if present)
            # "test10_character_0 (test10)" -> "test10_character_0"
            base_character_name = character_name.split(" (")[0] if " (" in character_name else character_name
            all_characters.add(base_character_name)
            character_to_video_mapping[base_character_name] = video_name
    
    print(f"\nFound {len(all_characters)} unique characters to process")
    
    # Process each character
    thumbnails_metadata = {}
    successful_thumbnails = 0
    
    for base_character_name in sorted(all_characters):
        print(f"\nProcessing character: {base_character_name}")
        
        # Find all detections for this character
        character_detections = []
        
        # Query face dataset for this character using the base name
        # Use filter_labels instead of match for nested detections
        try:
            view = face_dataset.filter_labels("character_detections", fo.ViewField("label") == base_character_name)
            print(f"  Found {len(view)} frames with this character")
        except Exception as e:
            print(f"  Error with filter_labels query: {e}")
            # Fallback to manual search if filter_labels fails
            view = face_dataset.view()
            print(f"  Using manual search as fallback...")
        
        # Continue with existing logic...
        # Collect all detections with quality scores
        for sample in view:
            if sample.has_field("character_detections"):
                # Load the actual image to get dimensions
                try:
                    img = cv2.imread(sample.filepath)
                    if img is None:
                        continue
                    frame_shape = img.shape[:2]  # (height, width)
                    
                    for detection in sample.character_detections.detections:
                        if detection.label == base_character_name:
                            # Calculate quality score
                            quality_score = calculate_quality_score(detection, frame_shape)
                            
                            character_detections.append({
                                "sample": sample,
                                "detection": detection,
                                "quality_score": quality_score,
                                "frame_shape": frame_shape
                            })
                            
                except Exception as e:
                    print(f"  Error processing {sample.filepath}: {e}")
                    continue
        
        # If we still have no detections and we used the full dataset view, 
        # it means the character doesn't exist in the current dataset
        if not character_detections and len(view) == len(face_dataset):
            print(f"  Character '{base_character_name}' not found in current dataset")
            continue
        
        if not character_detections:
            print(f"  No valid detections found for {base_character_name}")
            continue
        
        # Sort by quality score (highest first)
        character_detections.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Select the best detection
        best_detection = character_detections[0]
        print(f"  Best detection quality score: {best_detection['quality_score']:.3f}")
        
        # Load the source image
        source_sample = best_detection["sample"]
        source_image = cv2.imread(source_sample.filepath)
        
        if source_image is None:
            print(f"  Could not load source image: {source_sample.filepath}")
            continue
        
        # Crop face and create thumbnail
        try:
            bbox = best_detection["detection"].bounding_box
            
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
            
            # Crop face with padding
            thumbnail = crop_face_with_padding(pil_image, bbox, padding_ratio=0.3, target_size=(200, 200))
            
            # Enhance thumbnail
            thumbnail = enhance_thumbnail(thumbnail)
            
            # Generate filename (replace spaces and special characters)
            safe_character_name = base_character_name.replace(" ", "_").replace("(", "").replace(")", "")
            thumbnail_filename = f"{safe_character_name}.jpg"
            thumbnail_path = os.path.join(thumbnails_dir, thumbnail_filename)
            
            # Save thumbnail
            thumbnail.save(thumbnail_path, "JPEG", quality=90)
            
            # Store metadata
            thumbnails_metadata[base_character_name] = {
                "thumbnail_file": thumbnail_filename,
                "source_frame": source_sample.filepath,
                "confidence": float(best_detection["detection"].confidence) if hasattr(best_detection["detection"], 'confidence') else 0.5,
                "quality_score": best_detection["quality_score"],
                "bounding_box": list(bbox),
                "created_at": datetime.now().isoformat(),
                "total_detections": len(character_detections)
            }
            
            print(f"  ✓ Thumbnail saved: {thumbnail_filename}")
            successful_thumbnails += 1
            
        except Exception as e:
            print(f"  ✗ Error creating thumbnail for {base_character_name}: {e}")
            continue
    
    # Save metadata
    metadata_path = os.path.join(thumbnails_dir, "thumbnails_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(thumbnails_metadata, f, indent=2)
    
    print(f"\nThumbnail generation complete:")
    print(f"- Generated {successful_thumbnails} thumbnails successfully")
    print(f"- Processed {len(all_characters)} total characters")
    print(f"- Saved to: {thumbnails_dir}")
    print(f"- Metadata saved to: {metadata_path}")
    
    return successful_thumbnails > 0

if __name__ == "__main__":
    success = generate_character_thumbnails()
    if success:
        print("\n✅ Character thumbnails generated successfully!")
    else:
        print("\n❌ Failed to generate character thumbnails") 