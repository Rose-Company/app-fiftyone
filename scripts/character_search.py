import fiftyone as fo
from fiftyone import ViewField as F
import os
import numpy as np
import json
import time
from datetime import datetime, timedelta

# MongoDB connection configuration
database_uri = os.getenv("FIFTYONE_DATABASE_URI")
if not database_uri:
    print("Error: FIFTYONE_DATABASE_URI environment variable is required!")
    exit(1)
fo.config.database_uri = database_uri

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

def frame_to_timestamp(frame_number, fps=24):
    """Convert frame number to timestamp string"""
    # Frame numbering starts from 1, so subtract 1 before dividing by fps
    total_seconds = (frame_number - 1) / fps
    # Round to nearest second for proper display
    total_seconds = round(total_seconds)
    minutes, seconds = divmod(total_seconds, 60)
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

def detect_video_fps(face_dataset, video_name):
    """
    Detect FPS for a specific video by analyzing frame patterns
    
    Args:
        face_dataset: FiftyOne dataset
        video_name: Video name (e.g., "test1.mp4")
        
    Returns:
        int: Detected FPS
    """
    # Get video base name without extension
    video_base = video_name.replace('.mp4', '')
    
    # Find samples from this video - use better filtering
    video_samples = []
    for sample in face_dataset:
        sample_path = sample.filepath
        # Check if this sample belongs to the video
        if f"/{video_base}/" in sample_path or f"\\{video_base}\\" in sample_path:
            if hasattr(sample, 'frame_number'):
                video_samples.append(sample.frame_number)
    
    if len(video_samples) < 10:
        print(f"  Warning: Could not detect FPS for {video_name}, using default 24")
        return 24
    
    # Sort frame numbers
    video_samples.sort()
    
    # Analyze frame patterns to detect FPS
    # Common patterns: 1,25,49,73... (24fps), 1,31,61,91... (30fps), 1,26,51,76... (25fps)
    
    # Check for common FPS patterns
    if len(video_samples) >= 3:
        # Calculate intervals between consecutive frames
        intervals = []
        for i in range(1, min(10, len(video_samples))):  # Check more samples
            interval = video_samples[i] - video_samples[i-1]
            intervals.append(interval)
        
        # Find the most common interval (this is the FPS)
        from collections import Counter
        interval_counts = Counter(intervals)
        most_common_interval = interval_counts.most_common(1)[0][0]
        
        # The most common interval IS the FPS
        fps = most_common_interval
        
        # Validate against known FPS values
        if fps not in [24, 25, 30]:
            # If not a standard FPS, default to 24
            fps = 24
            
        print(f"  Detected FPS for {video_name}: {fps} (most common interval: {most_common_interval})")
        return fps
    
    print(f"  Warning: Could not detect FPS for {video_name}, using default 24")
    return 24

def merge_close_segments(segments, merge_threshold_seconds=2, fps=24):
    """
    Merge segments that are very close to each other
    
    Args:
        segments: List of segment dictionaries with start_frame, end_frame, etc.
        merge_threshold_seconds: Maximum gap in seconds to merge segments
        fps: Frames per second for the video
    
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
        
        # Use actual FPS for gap calculation
        gap_frames = next_segment["start_frame"] - current["end_frame"]
        gap_seconds = gap_frames / fps
        
        if gap_seconds <= merge_threshold_seconds:
            # Merge segments
            current["end_frame"] = next_segment["end_frame"]
            current["end_time"] = next_segment["end_time"]
            current["duration"] = current["end_frame"] - current["start_frame"]
            #print(f"    Merged segments: gap was {gap_seconds:.1f}s")
        else:
            # Add current segment and start new one
            merged.append(current)
            current = next_segment.copy()
    
    # Add the last segment
    merged.append(current)
    
    return merged

def group_consecutive_frames(frame_numbers, fps=24, max_gap_seconds=1):
    """
    Group consecutive frame numbers into scenes based on actual time gaps
    
    Args:
        frame_numbers: List of frame numbers
        fps: Frames per second for the video
        max_gap_seconds: Maximum gap in seconds to consider frames part of the same scene
    """
    if not frame_numbers:
        return []
    
    # Sắp xếp các frame và loại bỏ duplicates
    sorted_frames = sorted(list(set(frame_numbers)))
    
    if len(sorted_frames) == 1:
        return [(sorted_frames[0], sorted_frames[0])]
    
    # Tạo groups dựa trên khoảng cách thời gian thực tế
    groups = []
    current_start = sorted_frames[0]
    current_end = sorted_frames[0]
    
    for i in range(1, len(sorted_frames)):
        current_frame = sorted_frames[i]
        prev_frame = sorted_frames[i-1]
        
        # Tính thời gian thực tế: (frame - 1) / fps
        current_time = (current_frame - 1) / fps
        prev_time = (prev_frame - 1) / fps
        time_gap = current_time - prev_time
        
        # Nếu khoảng cách thời gian <= max_gap_seconds, coi là cùng scene
        if time_gap <= max_gap_seconds:
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
    print(f"  Frame grouping: {len(sorted_frames)} frames → {len(groups)} groups (max_gap={max_gap_seconds}s)")
    if len(groups) <= 5:  # Only show details for small number of groups
        for i, (start, end) in enumerate(groups):
            start_time = (start - 1) / fps
            end_time = (end - 1) / fps
            duration = end_time - start_time
            print(f"    Group {i+1}: frames {start}-{end} (time: {start_time:.1f}s-{end_time:.1f}s, duration: {duration:.1f}s)")
    
    return groups

def load_existing_character_index():
    """
    Load existing character index from individual video files
    """
    character_index = {}
    data_dir = "/fiftyone/data"
    
    # Load all existing video JSON files
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and not filename.startswith('character_'):
            # This is a video file (e.g., test1.json, test2.json)
            video_name = filename.replace('.json', '.mp4')
            file_path = os.path.join(data_dir, filename)
            
            try:
                with open(file_path, "r") as f:
                    character_index[video_name] = json.load(f)
            except:
                print(f"Warning: Could not load {filename}")
    
    return character_index

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
    if not fo.dataset_exists("video_dataset_final"):
        print("Video dataset not found. Please import videos first.")
        return False
        
    if not fo.dataset_exists("video_dataset_final_frames"):
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
    video_dataset = fo.load_dataset("video_dataset_final")
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
    
    # Detect FPS for each video
    print("\nDetecting FPS for each video...")
    video_fps_mapping = {}
    processed_videos = set()
    
    for character, videos in character_appearances.items():
        for video_name in videos.keys():
            if video_name not in processed_videos:
                fps = detect_video_fps(face_dataset, video_name)
                video_fps_mapping[video_name] = fps
                processed_videos.add(video_name)
    
    # Group consecutive frames into scenes for each character
    character_scenes = {}
    for character, videos in character_appearances.items():
        character_scenes[character] = {}
        for video_name, frames in videos.items():
            print(f"\nProcessing character '{character}' in video '{video_name}':")
            print(f"  Found {len(frames)} frame appearances")
            
            # Use actual FPS for time-based grouping
            fps = video_fps_mapping.get(video_name, 24)
            # Allow gap of 1 second between frames to be considered same scene
            max_gap_seconds = 1.0
            
            print(f"  Using FPS={fps}, max_gap={max_gap_seconds} seconds")
            character_scenes[character][video_name] = group_consecutive_frames(frames, fps=fps, max_gap_seconds=max_gap_seconds)
    
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
            
            # Use detected FPS instead of metadata
            fps = video_fps_mapping.get(video_name, 24)
            
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
            
            # Segments are already properly grouped by time-based logic
            print(f"  Created {len(scenes_list)} segments")
            
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
    
    # Save individual video JSON files
    print("\nSaving individual video JSON files...")
    data_dir = "/fiftyone/data"
    
    # Save each video as a separate JSON file
    for video_name, video_data in character_index.items():
        # Convert video name to filename (e.g., test1.mp4 -> test1.json)
        video_basename = os.path.splitext(video_name)[0]  # Remove .mp4
        json_filename = f"{video_basename}.json"
        json_path = os.path.join(data_dir, json_filename)
        
        with open(json_path, "w") as f:
            json.dump(video_data, f, indent=2)
        
        print(f"  Saved {json_filename}")
    
    # Create character update metadata file
    changed_videos = changes["new_videos"] + changes["updated_videos"]
    
    if changed_videos:
        print("\nCreating character_update_metadata.json...")
        
        # Save metadata file for tracking changes
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "changes": changes,
            "videos_count": len(changed_videos),
            "changed_files": [f"{os.path.splitext(v)[0]}.json" for v in changed_videos],
            "total_videos": len(character_index)
        }
        
        with open("/fiftyone/data/character_update_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Character update metadata contains {len(changed_videos)} changed videos:")
        for video_name in changed_videos:
            if video_name in character_index:
                status = "NEW" if video_name in changes["new_videos"] else "UPDATED"
                characters = character_index[video_name]
                json_filename = f"{os.path.splitext(video_name)[0]}.json"
                print(f"- {json_filename} ({status}): {len(characters)} characters")
                for character, scenes in characters.items():
                    print(f"  - {character}: {len(scenes)} appearances")
    
    print(f"\nCharacter indexing complete:")
    print(f"- Total videos: {len(character_index)}")
    print(f"- Changed videos: {len(changed_videos) if changed_videos else 0}")
    print(f"- Individual JSON files saved in /fiftyone/data/")
    
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
    # Load character index from individual video files
    character_index = load_existing_character_index()
    
    if not character_index:
        print("No character index files found. Please create the character index first.")
        return {}
    
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
    if not fo.dataset_exists("video_dataset_final"):
        print("Video dataset not found. Please import videos first.")
        return
    
    video_dataset = fo.load_dataset("video_dataset_final")
    
    # Get characters from the index
    character_index = load_existing_character_index()
    
    if not character_index:
        print("No character index files found. Please create the character index first.")
        return
    
    characters = sorted(list(character_index.keys()))
    
    # Create a view for search results
    search_view = create_character_search_view(video_dataset, characters)
    
    # Launch the app with the base dataset
    session = fo.launch_app(video_dataset)
    
    # Print instructions for using the search functionality
    print("\nCharacter search is ready!")
    print("Use the following steps to search for character appearances:")
    print("1. Open the FiftyOne App at http://localhost:" + os.getenv("FIFTYONE_PORT") + "")
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
    
    