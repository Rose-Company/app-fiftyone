import cv2
import numpy as np
import face_recognition
import fiftyone as fo
from fiftyone import ViewField as F
import gc
import os

def extract_face_from_bbox(image, bbox):
    """
    Trích xuất vùng khuôn mặt từ bounding box
    """
    # Chuyển đổi bbox từ [x, y, width, height] (tỷ lệ tương đối)
    # sang [top, right, bottom, left] (tọa độ pixel)
    height, width = image.shape[:2]
    x, y, w, h = bbox
    top = int(y * height)
    left = int(x * width)
    bottom = int((y + h) * height)
    right = int((x + w) * width)
    
    # Đảm bảo tọa độ nằm trong ảnh
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    left = max(0, left)
    
    # Kiểm tra nếu bounding box không hợp lệ
    if top >= bottom or left >= right:
        print(f"Bounding box không hợp lệ: {bbox}")
        return None
    
    # Trích xuất vùng khuôn mặt
    face_image = image[top:bottom, left:right]
    return face_image

def get_face_embedding(face_image, model="hog"):
    """
    Trích xuất embedding từ ảnh khuôn mặt
    """
    # Chuyển đổi sang RGB nếu cần
    if len(face_image.shape) == 3 and face_image.shape[2] == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Tìm vị trí khuôn mặt
    face_locations = face_recognition.face_locations(face_image, model=model)
    if not face_locations:
        return None
        
    # Trích xuất embedding
    face_encodings = face_recognition.face_encodings(face_image, face_locations)
    return face_encodings[0] if face_encodings else None

def extract_faces_from_frames():
    """
    Trích xuất khuôn mặt từ frames có người
    """
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Load dataset frames
    frames_dataset_name = "video_dataset_frames"
    if not fo.dataset_exists(frames_dataset_name):
        print(f"Không tìm thấy dataset '{frames_dataset_name}'")
        return
        
    frames_dataset = fo.load_dataset(frames_dataset_name)
    print(f"Đã load dataset '{frames_dataset_name}' với {len(frames_dataset)} frames")
    
    # Tạo dataset mới cho khuôn mặt nếu chưa tồn tại
    face_dataset_name = "video_dataset_faces_test"
    if fo.dataset_exists(face_dataset_name):
        face_dataset = fo.load_dataset(face_dataset_name)
        print(f"Dataset '{face_dataset_name}' đã tồn tại với {len(face_dataset)} samples")
    else:
        face_dataset = fo.Dataset(face_dataset_name)
        print(f"Đã tạo dataset mới '{face_dataset_name}'")
    
    # Lấy danh sách ID mẫu trong face_dataset để kiểm tra
    face_dataset_ids = set(face_dataset.values("id"))
    
    # Tìm các frames có người nhưng chưa có face_embeddings hoặc chưa nằm trong face_dataset
    query = (
        F("person_detections").exists() & 
        (
            (~F("face_embeddings").exists()) |
            (~F("id").is_in(list(face_dataset_ids)))
        )
    )
    unprocessed_frames = frames_dataset.match(query)
    total_unprocessed = len(unprocessed_frames)
    
    if total_unprocessed == 0:
        print("Không có frames mới cần xử lý trích xuất khuôn mặt.")
        return
    
    print(f"Tìm thấy {total_unprocessed} frames mới cần xử lý trích xuất khuôn mặt.")
    
    # Xử lý theo batch để tiết kiệm bộ nhớ
    batch_size = 10
    batch_count = (total_unprocessed + batch_size - 1) // batch_size
    total_faces = 0
    processed_frames = 0
    
    for i in range(batch_count):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_unprocessed)
        
        print(f"Xử lý batch {i+1}/{batch_count} (frames {start_idx}-{end_idx}/{total_unprocessed})...")
        
        # Lấy batch samples
        batch_ids = unprocessed_frames.values("id")[start_idx:end_idx]
        batch_view = frames_dataset.select(batch_ids)
        
        for sample in batch_view:
            # Đọc ảnh từ disk
            try:
                image = cv2.imread(sample.filepath)
                
                if image is None:
                    print(f"Không thể đọc ảnh từ {sample.filepath}")
                    continue
                
                # Chuyển từ BGR sang RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Nếu có person detections thì sử dụng nó, nếu không tìm mặt trực tiếp
                face_embeddings = []
                
                if sample.has_field("person_detections") and len(sample.person_detections.detections) > 0:
                    # Xử lý từ person detections
                    detections = sample.person_detections.detections
                    
                    for det in detections:
                        if det.label == "person":
                            # Trích xuất vùng khuôn mặt
                            person_crop = extract_face_from_bbox(image_rgb, det.bounding_box)
                            
                            if person_crop is None or person_crop.size == 0:
                                continue
                            
                            # Tìm và trích xuất khuôn mặt từ vùng người
                            embedding = get_face_embedding(person_crop, model="cnn")  # Sử dụng HOG cho tốc độ
                            
                            if embedding is not None:
                                face_embeddings.append({
                                    "embedding": embedding.tolist(),
                                    "bounding_box": det.bounding_box,
                                    "confidence": det.confidence
                                })
                                total_faces += 1
                else:
                    # Tìm khuôn mặt trực tiếp trên toàn bộ ảnh
                    face_locations = face_recognition.face_locations(image_rgb, model="hog")
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                        
                        for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
                            # Chuyển face_location thành bounding box [x, y, w, h]
                            top, right, bottom, left = location
                            height, width = image.shape[:2]
                            
                            x = left / width
                            y = top / height
                            w = (right - left) / width
                            h = (bottom - top) / height
                            
                            face_embeddings.append({
                                "embedding": encoding.tolist(),
                                "bounding_box": [x, y, w, h],
                                "confidence": 0.9  # Default confidence
                            })
                            total_faces += 1
                
                # Lưu thông tin khuôn mặt vào frame
                if face_embeddings:
                    sample["face_embeddings"] = face_embeddings
                    
                    # Đảm bảo thông tin video_id được lưu
                    if not sample.has_field("video_id") and hasattr(sample.metadata, "source_video_path"):
                        video_path = sample.metadata.source_video_path
                        video_name = os.path.basename(video_path)
                        video_id = os.path.splitext(video_name)[0]
                        sample["video_id"] = video_id
                    
                    sample.save()
                    processed_frames += 1
                    
                    # Thêm vào dataset khuôn mặt nếu chưa có
                    if sample.id not in face_dataset_ids:
                        face_dataset.add_sample(sample)
                        face_dataset_ids.add(sample.id)
                
                # Giải phóng bộ nhớ
                del image
                del image_rgb
                gc.collect()
                
            except Exception as e:
                print(f"Lỗi khi xử lý frame {sample.id}: {str(e)}")
                continue
        
        # Giải phóng bộ nhớ sau mỗi batch
        del batch_view
        gc.collect()
    
    # Thống kê kết quả
    frames_with_faces = len(face_dataset)
    frames_with_faces_new = processed_frames
    
    print(f"\nKết quả xử lý khuôn mặt:")
    print(f"- Số frames mới được xử lý: {total_unprocessed}")
    print(f"- Số frames mới có khuôn mặt: {frames_with_faces_new}")
    print(f"- Số khuôn mặt mới phát hiện được: {total_faces}")
    print(f"- Tổng số frames có khuôn mặt trong dataset: {frames_with_faces}")
    
    # Tránh division by zero
    if total_unprocessed > 0:
        face_ratio = frames_with_faces_new / total_unprocessed * 100
        print(f"- Tỷ lệ frames mới có khuôn mặt: {face_ratio:.2f}%")
    
    print("Truy cập http://localhost:5151 để xem kết quả")

if __name__ == "__main__":
    extract_faces_from_frames() 