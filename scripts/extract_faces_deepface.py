import cv2
import numpy as np
import fiftyone as fo
from fiftyone import ViewField as F
from deepface import DeepFace
import os
import gc
import time

# --- CONFIGURATION ---
# Model and Detector choice
# Recommended models: "ArcFace", "Facenet512", "VGG-Face"
# Recommended detectors: "retinaface", "mtcnn", "yunet"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"

# Dataset names
FRAMES_DATASET = "video_dataset_final_frames"
# Create a descriptive name for the output dataset
FACE_DATASET_NAME = f"video_dataset_faces_deepface_{MODEL_NAME.lower()}_{DETECTOR_BACKEND}_final"

# Video IDs to process - specify which videos to test
# Examples:
VIDEO_IDS_TO_PROCESS = ["test2", "test3", "test4", "test5"]  # Process only these specific videos
#VIDEO_IDS_TO_PROCESS = ["test1"]            # Process only one video
# VIDEO_IDS_TO_PROCESS = None                  # Process all videos (default behavior)
# VIDEO_IDS_TO_PROCESS = None  # Change this to specify which videos to process

# --- END CONFIGURATION ---

def process_frames_with_deepface():
    """
    Processes video frames to detect faces and extract embeddings using DeepFace.
    """
    print("="*60)
    print("=== Bắt đầu trích xuất khuôn mặt với DeepFace (Simplified) ===")
    print(f"Model: {MODEL_NAME} | Detector: {DETECTOR_BACKEND}")
    print("="*60)

    # Pre-load the model to avoid loading it for each frame
    try:
        print("Đang tải model và detector... (có thể mất vài phút lần đầu)")
        _ = DeepFace.build_model(MODEL_NAME)
        print("Model đã được tải thành công.")
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return

    # Connect to MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Load frames dataset
    if not fo.dataset_exists(FRAMES_DATASET):
        print(f"Không tìm thấy dataset frames '{FRAMES_DATASET}'")
        return
        
    frames_dataset = fo.load_dataset(FRAMES_DATASET)
    print(f"Đã load dataset '{FRAMES_DATASET}' với {len(frames_dataset)} frames")
    
    # Filter frames by video_id if specified
    if VIDEO_IDS_TO_PROCESS is not None:
        print(f"\nLọc frames theo video_id được chỉ định: {VIDEO_IDS_TO_PROCESS}")
        
        # Check which video_ids are available in the dataset
        available_video_ids = set()
        for sample in frames_dataset:
            if sample.has_field("video_id") and sample.video_id:
                available_video_ids.add(sample.video_id)
        
        print(f"Video IDs có sẵn trong dataset: {sorted(available_video_ids)}")
        
        # Validate requested video_ids
        requested_ids = set(VIDEO_IDS_TO_PROCESS)
        valid_ids = requested_ids.intersection(available_video_ids)
        invalid_ids = requested_ids - available_video_ids
        
        if invalid_ids:
            print(f"CẢNH BÁO: Các video_id không tồn tại: {sorted(invalid_ids)}")
        
        if not valid_ids:
            print("KHÔNG CÓ VIDEO_ID HỢP LỆ ĐỂ XỬ LÝ!")
            return
        
        print(f"Sẽ xử lý các video_id: {sorted(valid_ids)}")
        
        # Filter the dataset to only include specified video_ids
        frames_dataset = frames_dataset.match(F("video_id").is_in(list(valid_ids)))
        print(f"Sau khi lọc: {len(frames_dataset)} frames")
    else:
        print("Xử lý tất cả frames (không có lọc video_id)")
    
    # Create or load existing dataset for the faces
    if fo.dataset_exists(FACE_DATASET_NAME):
        face_dataset = fo.load_dataset(FACE_DATASET_NAME)
        print(f"Dataset '{FACE_DATASET_NAME}' đã tồn tại với {len(face_dataset)} samples.")
        
        # Get list of video_ids already processed
        processed_video_ids = set()
        for sample in face_dataset:
            if sample.has_field("video_id") and sample.video_id:
                processed_video_ids.add(sample.video_id)
        
        if processed_video_ids:
            print(f"Video IDs đã được xử lý trước đó: {sorted(processed_video_ids)}")
        
        # Filter out frames from videos that have already been processed
        if VIDEO_IDS_TO_PROCESS is not None:
            # Only process requested videos that haven't been processed yet
            videos_to_process = set(VIDEO_IDS_TO_PROCESS) - processed_video_ids
            if not videos_to_process:
                print(f"TẤT CẢ VIDEO IDs YÊU CẦU ĐÃ ĐƯỢC XỬ LÝ: {VIDEO_IDS_TO_PROCESS}")
                print("Sử dụng dataset hiện có.")
                return
            print(f"Chỉ xử lý video IDs chưa được xử lý: {sorted(videos_to_process)}")
            frames_dataset = frames_dataset.match(F("video_id").is_in(list(videos_to_process)))
        else:
            # Process all videos that haven't been processed yet
            if processed_video_ids:
                frames_dataset = frames_dataset.match(~F("video_id").is_in(list(processed_video_ids)))
            
        print(f"Frames còn lại cần xử lý: {len(frames_dataset)}")
        if len(frames_dataset) == 0:
            print("Không có frames mới cần xử lý!")
            return
            
    else:
        face_dataset = fo.Dataset(FACE_DATASET_NAME)
        face_dataset.persistent = True
        print(f"Đã tạo dataset mới '{FACE_DATASET_NAME}'")

    # Get all sample IDs to process
    sample_ids = frames_dataset.values("id")
    total_frames = len(sample_ids)
    
    batch_size = 50  # Process 50 frames at a time
    batch_count = (total_frames + batch_size - 1) // batch_size

    total_faces_embedded = 0
    start_time = time.time()

    print(f"\nBắt đầu xử lý {total_frames} frames trong {batch_count} batches (batch size: {batch_size})...")

    for i in range(batch_count):
        batch_start_time = time.time()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_frames)
        
        print(f"\n{'='*20} BATCH {i+1}/{batch_count} (frames {start_idx+1}-{end_idx}) {'='*20}")
        
        batch_ids = sample_ids[start_idx:end_idx]
        batch_view = frames_dataset.select(batch_ids)

        with fo.ProgressBar(total=len(batch_view)) as pb:
            for sample in batch_view.iter_samples(autosave=True, progress=pb):
                try:
                    img_path = sample.filepath

                    # Step 1: Detect faces and get their crops and locations
                    face_objects = DeepFace.extract_faces(
                        img_path=img_path,
                        detector_backend=DETECTOR_BACKEND,
                        enforce_detection=False,
                        align=True
                    )

                    if not face_objects or (len(face_objects) == 1 and face_objects[0]['confidence'] == 0):
                        continue

                    valid_face_embeddings = []
                    img_height, img_width = cv2.imread(img_path).shape[:2]

                    for face_obj in face_objects:
                        try:
                            # Step 2: Get embedding for each detected face
                            embedding_obj = DeepFace.represent(
                                img_path=face_obj['face'],
                                model_name=MODEL_NAME,
                                enforce_detection=False,
                                detector_backend='skip'
                            )
                            
                            if not embedding_obj:
                                continue
                            
                            embedding = embedding_obj[0]['embedding']
                            facial_area = face_obj['facial_area']

                            relative_bbox = [
                                facial_area['x'] / img_width,
                                facial_area['y'] / img_height,
                                facial_area['w'] / img_width,
                                facial_area['h'] / img_height,
                            ]

                            valid_face_embeddings.append({
                                "embedding": embedding,
                                "bounding_box": relative_bbox,
                                "confidence": face_obj['confidence']
                            })
                            total_faces_embedded += 1

                        except Exception as inner_e:
                            # This might happen if a face crop is invalid for the represent() function
                            continue

                    # If any valid faces were found, add them to the sample and save
                    if valid_face_embeddings:
                        sample["face_embeddings"] = valid_face_embeddings
                        
                        if "video_id" not in sample and hasattr(sample.metadata, "source_video_path"):
                             video_path = sample.metadata.source_video_path
                             video_name = os.path.basename(video_path)
                             sample["video_id"] = os.path.splitext(video_name)[0]

                        face_dataset.add_sample(sample.copy())

                    # Clean up memory
                    del face_objects
                    gc.collect()
                    
                except Exception as e:
                    print(f"Lỗi nghiêm trọng khi xử lý frame {sample.filepath}: {str(e)}")
                    continue
        
        batch_end_time = time.time()
        print(f"--> Batch {i+1} hoàn thành trong {batch_end_time - batch_start_time:.2f} giây.")

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("=== KẾT QUẢ TRÍCH XUẤT HOÀN TẤT ===")
    print(f"Tổng thời gian xử lý: {total_time:.2f} giây")
    print(f"Tốc độ xử lý: {total_frames / total_time if total_time > 0 else 0:.2f} frames/giây")
    print("-" * 30)
    
    # Show processing summary by video_id
    if VIDEO_IDS_TO_PROCESS is not None:
        print(f"Video IDs được yêu cầu xử lý: {VIDEO_IDS_TO_PROCESS}")
        
        # Count frames and faces by video_id in the result dataset
        video_stats = {}
        for sample in face_dataset:
            if sample.has_field("video_id") and sample.video_id:
                video_id = sample.video_id
                if video_id not in video_stats:
                    video_stats[video_id] = {"frames": 0, "faces": 0}
                video_stats[video_id]["frames"] += 1
                if sample.has_field("face_embeddings"):
                    video_stats[video_id]["faces"] += len(sample.face_embeddings)
        
        print("Thống kê tổng hợp trong dataset:")
        for video_id, stats in sorted(video_stats.items()):
            status = "✓ MỚI XỬ LÝ" if video_id in VIDEO_IDS_TO_PROCESS else "đã có sẵn"
            print(f"- {video_id}: {stats['frames']} frames, {stats['faces']} faces ({status})")
    else:
        print("Đã xử lý incremental cho tất cả videos")
    
    print("-" * 30)
    print(f"Tổng số frames có khuôn mặt trong dataset: {len(face_dataset)}")
    total_faces_in_dataset = sum(len(sample.face_embeddings) for sample in face_dataset if sample.has_field("face_embeddings"))
    print(f"Tổng số khuôn mặt trong dataset: {total_faces_in_dataset}")
    print(f"Số khuôn mặt mới được trích xuất trong lần chạy này: {total_faces_embedded}")
    print("-" * 30)
    print(f"Dữ liệu khuôn mặt đã được lưu vào dataset: '{FACE_DATASET_NAME}'")
    print("\nTruy cập http://localhost:5151 để xem kết quả")
    print("="*60)


if __name__ == "__main__":
    process_frames_with_deepface() 