import cv2
import numpy as np
import face_recognition
import fiftyone as fo
from fiftyone import ViewField as F
import gc
import os
import argparse
import datetime

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

def extract_faces_for_video(video_name, reprocess=False, create_new_dataset=False, force=False):
    """
    Trích xuất khuôn mặt từ frames có người cho một video cụ thể
    
    Args:
        video_name: Tên video (không có đuôi mở rộng, ví dụ: 'test', 'test2')
        reprocess: Nếu True, xử lý lại tất cả frames dù đã có face_embeddings
        create_new_dataset: Nếu True, tạo dataset mới thay vì ghi đè lên dataset cũ
        force: Nếu True, bỏ qua xác nhận khi xử lý lại dữ liệu
    """
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Load dataset frames
    frames_dataset_name = "video_dataset_frames"
    if not fo.dataset_exists(frames_dataset_name):
        print(f"Không tìm thấy dataset '{frames_dataset_name}'")
        return False
        
    frames_dataset = fo.load_dataset(frames_dataset_name)
    print(f"Đã load dataset '{frames_dataset_name}' với {len(frames_dataset)} frames")
    
    # Xác nhận khi xử lý lại dữ liệu
    if reprocess and not force:
        confirmation = input(f"CẢNH BÁO: Bạn sắp xử lý lại dữ liệu cho video '{video_name}'. Điều này có thể ghi đè lên dữ liệu đã có. Tiếp tục? (y/n): ")
        if confirmation.lower() != 'y':
            print("Đã hủy quá trình xử lý.")
            return False
    
    # Tạo dataset cho khuôn mặt
    face_dataset_name = "video_dataset_faces"
    if create_new_dataset:
        # Tạo tên dataset mới với timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        face_dataset_name = f"video_dataset_faces_{video_name}_{timestamp}"
        face_dataset = fo.Dataset(face_dataset_name)
        print(f"Đã tạo dataset mới '{face_dataset_name}'")
    else:
        # Sử dụng dataset hiện có hoặc tạo mới nếu chưa tồn tại
        if fo.dataset_exists(face_dataset_name):
            face_dataset = fo.load_dataset(face_dataset_name)
            print(f"Dataset '{face_dataset_name}' đã tồn tại với {len(face_dataset)} samples")
        else:
            face_dataset = fo.Dataset(face_dataset_name)
            print(f"Đã tạo dataset mới '{face_dataset_name}'")
    
    # Lấy danh sách ID mẫu trong face_dataset để kiểm tra
    face_dataset_ids = set(face_dataset.values("id"))
    
    # Tạo pattern để tìm frames thuộc video cụ thể
    pattern = f"/frames/{video_name}/"
    print(f"Đang tìm frames của video '{video_name}'...")
    
    # Tính toán danh sách frames cần xử lý thủ công để tránh lỗi MongoDB
    frame_ids = []
    for sample in frames_dataset:
        if pattern in sample.filepath:
            if reprocess or not sample.has_field("face_embeddings"):
                # Chỉ xử lý frames có person_detections
                if sample.has_field("person_detections"):
                    frame_ids.append(sample.id)
    
    # Nếu không có frames nào cần xử lý thì kết thúc
    if not frame_ids:
        print(f"Không tìm thấy frames {'' if reprocess else 'mới '} cần xử lý trích xuất khuôn mặt cho video '{video_name}'")
        return True
    
    # Lọc các frames cần xử lý
    unprocessed_frames = frames_dataset.select(frame_ids)
    total_unprocessed = len(frame_ids)
    
    print(f"Tìm thấy {total_unprocessed} frames {'' if reprocess else 'mới '} cần xử lý trích xuất khuôn mặt cho video '{video_name}'")
    
    # Backup face_embeddings hiện có nếu đang xử lý lại
    if reprocess and not create_new_dataset:
        print("Đang lưu backup cho face_embeddings hiện có...")
        for sample in unprocessed_frames:
            if sample.has_field("face_embeddings"):
                sample["face_embeddings_backup"] = sample.face_embeddings
        frames_dataset.save()
        print("Đã lưu backup thành công.")
    
    # Xử lý theo batch để tiết kiệm bộ nhớ
    batch_size = 10
    batch_count = (total_unprocessed + batch_size - 1) // batch_size
    total_faces = 0
    processed_frames = 0
    
    for i in range(batch_count):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_unprocessed)
        
        if start_idx >= len(frame_ids):
            break
            
        # Lấy batch IDs từ danh sách đã tính trước
        current_batch_ids = frame_ids[start_idx:end_idx]
        
        if not current_batch_ids:
            continue
        
        print(f"Xử lý batch {i+1}/{batch_count} (frames {start_idx}-{end_idx}/{total_unprocessed})...")
        
        # Lấy batch samples
        batch_view = frames_dataset.select(current_batch_ids)
        
        try:
            for sample in batch_view:
                # Đọc ảnh từ disk
                try:
                    image = cv2.imread(sample.filepath)
                    
                    if image is None:
                        print(f"Không thể đọc ảnh từ {sample.filepath}")
                        continue
                    
                    # Chuyển từ BGR sang RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Xử lý từ person detections
                    face_embeddings = []
                    
                    if sample.has_field("person_detections") and len(sample.person_detections.detections) > 0:
                        detections = sample.person_detections.detections
                        
                        for det in detections:
                            if det.label == "person":
                                # Trích xuất vùng khuôn mặt
                                person_crop = extract_face_from_bbox(image_rgb, det.bounding_box)
                                
                                if person_crop is None or person_crop.size == 0:
                                    continue
                                
                                # Tìm và trích xuất khuôn mặt từ vùng người
                                embedding = get_face_embedding(person_crop, model="hog")  # Sử dụng HOG cho tốc độ
                                
                                if embedding is not None:
                                    face_embeddings.append({
                                        "embedding": embedding.tolist(),
                                        "bounding_box": det.bounding_box,
                                        "confidence": det.confidence
                                    })
                                    total_faces += 1
                    
                    # Lưu thông tin khuôn mặt vào frame
                    if face_embeddings:
                        # Tạo bản sao sample nếu đang tạo dataset mới
                        if create_new_dataset:
                            # Không lưu face_embeddings vào dataset gốc nếu tạo dataset mới
                            sample_for_face_dataset = sample.copy()
                            sample_for_face_dataset["face_embeddings"] = face_embeddings
                            
                            # Đảm bảo thông tin video_id được lưu
                            if not sample_for_face_dataset.has_field("video_id"):
                                sample_for_face_dataset["video_id"] = video_name
                            
                            # Thêm vào dataset khuôn mặt mới
                            face_dataset.add_sample(sample_for_face_dataset)
                        else:
                            # Lưu trực tiếp vào frame gốc
                            sample["face_embeddings"] = face_embeddings
                            
                            # Đảm bảo thông tin video_id được lưu
                            if not sample.has_field("video_id"):
                                sample["video_id"] = video_name
                            
                            sample.save()
                            
                            # Thêm vào dataset khuôn mặt nếu chưa có
                            if sample.id not in face_dataset_ids and not create_new_dataset:
                                face_dataset.add_sample(sample)
                                face_dataset_ids.add(sample.id)
                        
                        processed_frames += 1
                    
                    # Giải phóng bộ nhớ
                    del image
                    del image_rgb
                    
                except Exception as e:
                    print(f"Lỗi khi xử lý frame {sample.id}: {str(e)}")
                    continue
        except Exception as e:
            print(f"Lỗi khi xử lý batch {i+1}: {e}")
        
        # Giải phóng bộ nhớ sau mỗi batch
        del batch_view
        gc.collect()
    
    # Tính toán thống kê cho việc hiển thị
    frames_with_faces = 0
    total_video_frames = 0
    
    print("Đang tính toán thống kê...")
    for sample in frames_dataset:
        if pattern in sample.filepath:
            total_video_frames += 1
            if sample.has_field("face_embeddings") and sample.face_embeddings is not None and len(sample.face_embeddings) > 0:
                frames_with_faces += 1
    
    print(f"\nKết quả xử lý khuôn mặt cho video '{video_name}':")
    print(f"- Dataset được sử dụng: '{face_dataset_name}'")
    print(f"- Số frames được xử lý: {total_unprocessed}")
    print(f"- Số frames có khuôn mặt: {processed_frames}")
    print(f"- Số khuôn mặt phát hiện được: {total_faces}")
    if total_video_frames > 0:
        print(f"- Tổng số frames có khuôn mặt trong video này: {frames_with_faces}/{total_video_frames} ({frames_with_faces/total_video_frames*100:.2f}%)")
    else:
        print(f"- Tổng số frames có khuôn mặt trong video này: {frames_with_faces}/{total_video_frames} (0.00%)")
    print("Truy cập http://localhost:5151 để xem kết quả")
    
    return True

if __name__ == "__main__":
    # Tạo parser cho tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Extract faces from frames of a specific video")
    parser.add_argument("video_name", help="Name of the video to process (e.g., 'test', 'test2')")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess all frames, even if they already have face_embeddings")
    parser.add_argument("--new-dataset", action="store_true", help="Create a new dataset instead of overwriting the existing one")
    parser.add_argument("--force", action="store_true", help="Skip confirmation when reprocessing data")
    
    # Parse tham số
    args = parser.parse_args()
    
    # Chạy hàm xử lý với video được chỉ định
    extract_faces_for_video(args.video_name, args.reprocess, args.new_dataset, args.force) 