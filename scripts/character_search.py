import fiftyone as fo
from fiftyone import ViewField as F
import os
import numpy as np
import json
import time
from datetime import datetime, timedelta
import cv2
import face_recognition

# MongoDB connection configuration
fo.config.database_uri = "mongodb://mongo:27017"

def load_character_dataset():
    """
    Load or create the character dataset with temporal detections
    """
    # Check if the character dataset exists
    character_dataset_name = "video_characters"
    if fo.dataset_exists(character_dataset_name):
        print(f"Loading existing character dataset '{character_dataset_name}'")
        return fo.load_dataset(character_dataset_name)
    
    # Create a new character dataset if it doesn't exist
    print(f"Creating new character dataset '{character_dataset_name}'")
    return fo.Dataset(character_dataset_name)

def frame_to_timestamp(frame_number, fps=1):
    """Convert frame number to timestamp string"""
    seconds = frame_number / fps
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def timestamp_to_frame(timestamp, fps=1):
    """Convert timestamp string to frame number"""
    h, m, s = map(int, timestamp.split(':'))
    total_seconds = h * 3600 + m * 60 + s
    return int(total_seconds * fps)

def extract_video_name_from_filepath(filepath):
    """
    Trích xuất tên video từ đường dẫn của frame
    Ví dụ: "/fiftyone/data/frames/test2/000217.jpg" -> "test2.mp4"
    """
    # Lấy tên thư mục chứa frames (thường là tên video không có extension)
    parts = filepath.split('/')
    for i, part in enumerate(parts):
        if part == "frames" and i < len(parts) - 1:
            # Lấy tên thư mục sau "frames"
            video_name = parts[i+1]
            return f"{video_name}.mp4"
    
    # Nếu không thể xác định từ đường dẫn
    return os.path.basename(filepath)

def merge_close_segments(segments, merge_threshold_seconds=2):
    """
    Merge segments that are very close to each other
    
    Args:
        segments: List of segment dictionaries with start_frame, end_frame, etc.
        merge_threshold_seconds: Maximum gap in seconds to merge segments
    
    Returns:
        List of merged segments
    """
    if len(segments) <= 1:
        return segments
    
    # Sort by start_frame
    sorted_segments = sorted(segments, key=lambda x: x["start_frame"])
    
    merged = []
    current = sorted_segments[0].copy()
    
    for i in range(1, len(sorted_segments)):
        next_segment = sorted_segments[i]
        
        # Assume 24 fps for gap calculation
        gap_frames = next_segment["start_frame"] - current["end_frame"]
        gap_seconds = gap_frames / 24.0
        
        if gap_seconds <= merge_threshold_seconds:
            # Merge segments
            current["end_frame"] = next_segment["end_frame"]
            current["end_time"] = next_segment["end_time"]
            current["duration"] = current["end_frame"] - current["start_frame"]
            print(f"    Merged segments: gap was {gap_seconds:.1f}s")
        else:
            # Add current segment and start new one
            merged.append(current)
            current = next_segment.copy()
    
    # Add the last segment
    merged.append(current)
    
    return merged

def group_consecutive_frames(frame_numbers, max_gap=1):
    """
    Group consecutive frame numbers into scenes with improved gap handling
    
    Args:
        frame_numbers: List of frame numbers
        max_gap: Maximum gap between frames to consider them part of the same scene
                 Default is 1 frame (truly consecutive)
    """
    if not frame_numbers:
        return []
    
    # Sắp xếp các frame và loại bỏ duplicates
    sorted_frames = sorted(list(set(frame_numbers)))
    
    if len(sorted_frames) == 1:
        return [(sorted_frames[0], sorted_frames[0])]
    
    # Gộp frames với logic cải tiến
    groups = []
    current_start = sorted_frames[0]
    current_end = sorted_frames[0]
    
    for i in range(1, len(sorted_frames)):
        current_frame = sorted_frames[i]
        prev_frame = sorted_frames[i-1]
        
        # Nếu frame hiện tại gần với frame trước đó (trong khoảng max_gap)
        if current_frame <= prev_frame + max_gap:
            # Mở rộng nhóm hiện tại
            current_end = current_frame
        else:
            # Kết thúc nhóm hiện tại và bắt đầu nhóm mới
            groups.append((current_start, current_end))
            current_start = current_frame
            current_end = current_frame
    
    # Thêm nhóm cuối cùng
    groups.append((current_start, current_end))
    
    # Debug information
    print(f"  Frame grouping: {len(sorted_frames)} frames → {len(groups)} groups")
    if len(groups) <= 5:  # Only show details for small number of groups
        for i, (start, end) in enumerate(groups):
            duration_frames = end - start
            print(f"    Group {i+1}: frames {start}-{end} (duration: {duration_frames} frames)")
    
    return groups

def load_existing_character_index():
    """
    Load existing character index if it exists
    """
    index_path = "/fiftyone/data/character_index.json"
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                return json.load(f)
        except:
            print("Warning: Could not load existing character_index.json")
    return {}

def detect_changes(old_index, new_index):
    """
    Detect what has changed between old and new character index
    
    Returns:
        dict: Changes detected with structure:
        {
            "new_videos": [list of new video names],
            "updated_videos": [list of updated video names],
            "removed_videos": [list of removed video names],
            "has_changes": bool
        }
    """
    changes = {
        "new_videos": [],
        "updated_videos": [],
        "removed_videos": [],
        "has_changes": False
    }
    
    # Find new videos
    for video_name in new_index:
        if video_name not in old_index:
            changes["new_videos"].append(video_name)
            changes["has_changes"] = True
    
    # Find removed videos
    for video_name in old_index:
        if video_name not in new_index:
            changes["removed_videos"].append(video_name)
            changes["has_changes"] = True
    
    # Find updated videos (same video but different characters/scenes)
    for video_name in new_index:
        if video_name in old_index:
            # Compare character data
            old_characters = old_index[video_name]
            new_characters = new_index[video_name]
            
            # Check if character data is different
            if old_characters != new_characters:
                changes["updated_videos"].append(video_name)
                changes["has_changes"] = True
    
    return changes

def create_character_index():
    """
    Create a searchable index of characters with their appearances in videos
    Only creates new files if there are changes detected
    """
    # Load the necessary datasets
    if not fo.dataset_exists("video_dataset"):
        print("Video dataset not found. Please import videos first.")
        return False
        
    if not fo.dataset_exists("video_dataset_frames"):
        print("Frames dataset not found. Please extract frames first.")
        return False
    
    # Load the face dataset
    face_dataset_name = "video_dataset_faces_deepface_arcface_retinaface_final"
    if not fo.dataset_exists(face_dataset_name):
        print(f"Face dataset '{face_dataset_name}' not found. Please extract faces first.")
        return False
    
    face_dataset = fo.load_dataset(face_dataset_name)
    
    # Load existing character index for comparison
    print("Loading existing character index for comparison...")
    old_character_index = load_existing_character_index()
    
    # Get the original video dataset to create temporal annotations
    video_dataset = fo.load_dataset("video_dataset")
    if len(video_dataset) == 0:
        print("No videos found in the dataset.")
        return False
    
    # Create or load character dataset
    character_dataset = load_character_dataset()
    
    # Check if character grouping has been done
    if not face_dataset.has_sample_field("character_detections"):
        print("Character grouping has not been done yet. Please run group_faces_characters.py first.")
        return False
    
    # Get the mapping of characters to their appearances
    character_appearances = {}
    
    # Create mapping of frame filepath -> video name for faster lookup
    video_name_mapping = {}
    for video_sample in video_dataset:
        video_path = video_sample.filepath
        video_name = os.path.basename(video_path)
        folder_name = os.path.splitext(video_name)[0]  # Remove extension
        video_name_mapping[folder_name] = video_name
    
    # Process the dataset in batches to avoid memory issues
    batch_size = 50
    total_samples = len(face_dataset)
    batch_count = (total_samples + batch_size - 1) // batch_size
    
    print(f"Processing {total_samples} samples in {batch_count} batches...")
    
    for i in range(batch_count):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        
        print(f"Processing batch {i+1}/{batch_count} (samples {start_idx}-{end_idx}/{total_samples})...")
        
        # Get batch samples
        batch_ids = face_dataset.values("id")[start_idx:end_idx]
        batch_view = face_dataset.select(batch_ids)
        
        for sample in batch_view:
            if sample.has_field("character_detections") and hasattr(sample, "frame_number"):
                for detection in sample.character_detections.detections:
                    # Character ID is in the format "Character X" where X is the character ID
                    character_name = detection.label
                    
                    # Lấy tên video từ filepath của frame
                    frame_path = sample.filepath
                    video_name = extract_video_name_from_filepath(frame_path)
                    
                    frame_number = sample.frame_number
                    
                    if character_name not in character_appearances:
                        character_appearances[character_name] = {}
                    
                    if video_name not in character_appearances[character_name]:
                        character_appearances[character_name][video_name] = []
                        
                    character_appearances[character_name][video_name].append(frame_number)
    
    # Group consecutive frames into scenes for each character
    character_scenes = {}
    for character, videos in character_appearances.items():
        character_scenes[character] = {}
        for video_name, frames in videos.items():
            print(f"\nProcessing character '{character}' in video '{video_name}':")
            print(f"  Found {len(frames)} frame appearances")
            
            # Sử dụng max_gap nhỏ hơn cho frames thực sự liên tiếp
            # max_gap=1: chỉ gộp frames liên tiếp trực tiếp
            # max_gap=5: cho phép gộp frames cách nhau 1-5 frames (khoảng 0.2 giây)
            character_scenes[character][video_name] = group_consecutive_frames(frames, max_gap=5)
    
    # Create temporal detections for each video sample
    for video_sample in video_dataset:
        video_name = os.path.basename(video_sample.filepath)
        fps = video_sample.metadata.frame_rate if hasattr(video_sample.metadata, "frame_rate") else 24  # Default 24fps
        
        # Add temporal detections for each character
        for character, videos in character_scenes.items():
            if video_name in videos:
                # Create temporal detections for this character
                temporal_detections = []
                for start_frame, end_frame in videos[video_name]:
                    start_time = start_frame / fps
                    end_time = end_frame / fps
                    
                    temporal_detections.append(
                        fo.TemporalDetection(
                            label=character,
                            support=[start_time, end_time],
                            confidence=1.0
                        )
                    )
                
                # Add to the video sample
                field_name = f"character_{character.replace(' ', '_')}"
                video_sample[field_name] = fo.TemporalDetections(detections=temporal_detections)
                
        # Save the updated sample
        video_sample.save()
    
    # Create a JSON file with character index for the UI
    character_index = {}
    
    # Tạo cấu trúc video trước, sau đó mới đến nhân vật
    for character, videos in character_scenes.items():
        for video_name, scenes in videos.items():
            # Tạo entry cho video nếu chưa tồn tại
            if video_name not in character_index:
                character_index[video_name] = {}
            
            video_base_name = video_name.split(".")[0]  # Bỏ extension
            fps = 24  # Default fps
            
            # Tìm video tương ứng để lấy fps
            for video_sample in video_dataset:
                if video_base_name in video_sample.filepath:
                    fps = video_sample.metadata.frame_rate if hasattr(video_sample.metadata, "frame_rate") else 24
                    break
            
            # Tạo character_name có chứa tên video để phân biệt rõ hơn
            # Ví dụ: "Character 0 (test2)" thay vì chỉ "Character 0"
            character_name = character
            if not character_name.endswith(f"({video_base_name})"):
                character_name = f"{character} ({video_base_name})"
            
            # Thêm scenes vào character trong video này
            scenes_list = [
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": frame_to_timestamp(start_frame, fps),
                    "end_time": frame_to_timestamp(end_frame, fps),
                    "duration": end_frame - start_frame
                }
                for start_frame, end_frame in scenes
            ]
            
            # Merge close segments to fix fragmented appearances
            print(f"  Before merging: {len(scenes_list)} segments")
            scenes_list = merge_close_segments(scenes_list, merge_threshold_seconds=3)
            print(f"  After merging: {len(scenes_list)} segments")
            
            character_index[video_name][character_name] = scenes_list
    
    # Detect changes compared to existing index
    print("\nDetecting changes...")
    changes = detect_changes(old_character_index, character_index)
    
    if not changes["has_changes"]:
        print("No changes detected. Character index is up to date.")
        return True
    
    # Display detected changes
    print(f"\nChanges detected:")
    if changes["new_videos"]:
        print(f"- New videos: {changes['new_videos']}")
    if changes["updated_videos"]:
        print(f"- Updated videos: {changes['updated_videos']}")
    if changes["removed_videos"]:
        print(f"- Removed videos: {changes['removed_videos']}")
    
    # Save the updated character index
    print("\nSaving updated character index...")
    with open("/fiftyone/data/character_index.json", "w") as f:
        json.dump(character_index, f, indent=2)
    
    # Create character update file with only changed data
    character_update = {}
    
    # Add new and updated videos to the update file
    for video_name in changes["new_videos"] + changes["updated_videos"]:
        if video_name in character_index:
            character_update[video_name] = character_index[video_name]
    
    # Save the character update file
    if character_update:
        print("Creating character_update.json with changed data...")
        
        # Save with same structure as character_index.json for easy synchronization
        with open("/fiftyone/data/character_update.json", "w") as f:
            json.dump(character_update, f, indent=2)
        
        # Also save a metadata file for tracking changes
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "changes": changes,
            "videos_count": len(character_update)
        }
        
        with open("/fiftyone/data/character_update_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Character update file contains {len(character_update)} videos:")
        for video_name, characters in character_update.items():
            status = "NEW" if video_name in changes["new_videos"] else "UPDATED"
            print(f"- {video_name} ({status}): {len(characters)} characters")
            for character, scenes in characters.items():
                print(f"  - {character}: {len(scenes)} appearances")
    
    print(f"\nCharacter indexing complete:")
    print(f"- Total videos: {len(character_index)}")
    print(f"- Changed videos: {len(character_update)}")
    
    return True

def search_characters(target_characters, excluded_characters=None, min_duration=0):
    """
    Search for segments where specified characters appear but excluded characters don't
    
    Args:
        target_characters: List of character names to search for
        excluded_characters: List of character names to exclude
        min_duration: Minimum scene duration in seconds
        
    Returns:
        Dictionary of matching scenes
    """
    # Load character index
    index_path = "/fiftyone/data/character_index.json"
    if not os.path.exists(index_path):
        print("Character index not found. Please create the character index first.")
        return {}
    
    with open(index_path, "r") as f:
        character_index = json.load(f)
    
    # Initialize results
    matching_scenes = {}
    
    # For each video, check for target characters
    for video_name, characters in character_index.items():
        matching_scenes[video_name] = []
        
        # Check if target characters exist in this video
        for character in target_characters:
            if character in characters:
                for scene in characters[character]:
                    # Check duration
                    duration = scene["end_frame"] - scene["start_frame"]
                    if duration < min_duration:
                        continue
                        
                    # Check for excluded characters
                    exclude_scene = False
                    if excluded_characters:
                        for excluded in excluded_characters:
                            if excluded in characters:
                                for ex_scene in characters[excluded]:
                                    # Check if the excluded character appears in this scene
                                    if (scene["start_frame"] <= ex_scene["end_frame"] and 
                                        scene["end_frame"] >= ex_scene["start_frame"]):
                                        exclude_scene = True
                                        break
                    
                    if not exclude_scene:
                        matching_scenes[video_name].append({
                            "character": character,
                            "start_frame": scene["start_frame"],
                            "end_frame": scene["end_frame"],
                            "start_time": scene["start_time"],
                            "end_time": scene["end_time"],
                            "duration": duration
                        })
        
        # Remove empty videos
        if not matching_scenes[video_name]:
            del matching_scenes[video_name]
        else:
            # Sort scenes by start time
            matching_scenes[video_name].sort(key=lambda x: x["start_frame"])
    
    return matching_scenes

def create_character_search_view(video_dataset, characters):
    """
    Create a view with text fields to display character search results
    """
    # Create a view to add the search results to
    view = video_dataset.view()
    
    # Add selected character information to each sample
    for sample in view:
        sample["character_search"] = json.dumps([])
        sample.save()
    
    # Tag the view to identify it
    view = view.tag_samples("character_search_results")
    
    return view

def launch_character_search_app():
    """
    Launch FiftyOne app with character search functionality
    """
    # Check if we have the necessary data
    if not fo.dataset_exists("video_dataset"):
        print("Video dataset not found. Please import videos first.")
        return
    
    video_dataset = fo.load_dataset("video_dataset")
    
    # Get characters from the index
    index_path = "/fiftyone/data/character_index.json"
    if not os.path.exists(index_path):
        print("Character index not found. Please create the character index first.")
        return
    
    with open(index_path, "r") as f:
        character_index = json.load(f)
    
    characters = sorted(list(character_index.keys()))
    
    # Create a view for search results
    search_view = create_character_search_view(video_dataset, characters)
    
    # Launch the app with the base dataset
    session = fo.launch_app(video_dataset)
    
    # Print instructions for using the search functionality
    print("\nCharacter search is ready!")
    print("Use the following steps to search for character appearances:")
    print("1. Open the FiftyOne App at http://localhost:5151")
    print("2. Select the characters to search for and filter options")
    print("3. Use the 'video_dataset' to view the original videos")
    print(f"4. Characters available for search: {', '.join(characters)}")
    
    # Keep the session alive (important for Windows users)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCharacter search app closed.")
    
if __name__ == "__main__":
    # Create the character index
    index_created = create_character_index()
    
    