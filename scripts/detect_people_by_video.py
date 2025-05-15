import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import gc
import os
import cv2
import numpy as np
import argparse

# Copy hàm detect_people_with_opencv từ file detect_people.py
def detect_people_with_opencv(image_path, confidence_threshold=0.5):
    """
    Phát hiện người trong ảnh sử dụng OpenCV DNN và mô hình YOLO
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    height, width = image.shape[:2]
    
    # Tải mô hình YOLO nếu chưa được tải
    if not hasattr(detect_people_with_opencv, "net"):
        # Xác định đường dẫn đến thư mục mô hình
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Tải các file cấu hình và trọng số nếu chưa có
        cfg_path = os.path.join(model_dir, "yolov3.cfg")
        weights_path = os.path.join(model_dir, "yolov3.weights")
        classes_path = os.path.join(model_dir, "coco.names")
        
        # Tải mô hình nếu chưa có
        if not (os.path.exists(cfg_path) and os.path.exists(weights_path)):
            print("Đang tải mô hình YOLO...")
            # Tải từ URL công khai
            cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
            weights_url = "https://pjreddie.com/media/files/yolov3.weights"
            
            os.system(f"curl -o {cfg_path} {cfg_url}")
            os.system(f"curl -o {weights_path} {weights_url}")
            
            # Tải file classes
            if not os.path.exists(classes_path):
                os.system(f"curl -o {classes_path} https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
        
        # Đọc danh sách classes
        with open(classes_path, "r") as f:
            detect_people_with_opencv.classes = f.read().strip().split("\n")
        
        # Tải mô hình
        detect_people_with_opencv.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        
        # Đặt backend và target
        detect_people_with_opencv.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        detect_people_with_opencv.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Lấy các layer đầu ra
        layer_names = detect_people_with_opencv.net.getLayerNames()
        detect_people_with_opencv.output_layers = [layer_names[i - 1] for i in detect_people_with_opencv.net.getUnconnectedOutLayers()]
    
    # Chuẩn bị ảnh cho mô hình
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    detect_people_with_opencv.net.setInput(blob)
    
    # Thực hiện dự đoán
    outputs = detect_people_with_opencv.net.forward(detect_people_with_opencv.output_layers)
    
    # Xử lý kết quả
    boxes = []
    confidences = []
    class_ids = []
    
    # Xử lý mỗi đầu ra
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Chỉ lấy người (class_id = 0 cho "person" trong COCO dataset)
            if class_id == 0 and confidence > confidence_threshold:
                # Tọa độ được chuẩn hóa, nhân với chiều rộng và chiều cao để có tọa độ thực
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Tọa độ góc trên bên trái
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Chuyển đổi sang định dạng FiftyOne [x, y, width, height] (chuẩn hóa về khoảng [0,1])
                x_norm = max(0, x) / width
                y_norm = max(0, y) / height
                w_norm = min(width - x, w) / width
                h_norm = min(height - y, h) / height
                
                boxes.append([x_norm, y_norm, w_norm, h_norm])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences

def detect_people_for_video(video_name, reprocess=False):
    """
    Hàm xử lý phát hiện người cho frames của một video cụ thể
    
    Args:
        video_name: Tên video (không có đuôi mở rộng, ví dụ: 'test', 'test2')
        reprocess: Nếu True, xử lý lại tất cả frames dù đã có person_detections
    """
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"

    # Load dataset frames
    frames_dataset_name = "video_dataset_frames"
    if not fo.dataset_exists(frames_dataset_name):
        print(f"Không tìm thấy dataset '{frames_dataset_name}'")
        return False

    frames = fo.load_dataset(frames_dataset_name)
    print(f"Đã load dataset '{frames_dataset_name}' với {len(frames)} frames")

    # Tạo filter để lấy frames của video cụ thể
    pattern = f"/frames/{video_name}/"
    filter_expr = F("filepath").re_match(f".*{pattern}.*")
    
    # Nếu reprocess=False, chỉ xử lý frames chưa có person_detections
    if not reprocess:
        filter_expr = filter_expr & (~F("person_detections").exists())
    
    # Lấy frames từ video cụ thể
    video_frames = frames.match(filter_expr)
    
    # Lấy danh sách các frames thuộc video này để đảm bảo kết quả chính xác 
    print(f"Đang tìm frames của video '{video_name}'...")
    
    # Tính toán số lượng frames thủ công để tránh lỗi MongoDB
    frame_ids = []
    for sample in frames:
        if pattern in sample.filepath and (reprocess or not sample.has_field("person_detections")):
            frame_ids.append(sample.id)
    
    # Nếu không có frames nào cần xử lý thì kết thúc
    if not frame_ids:
        print(f"Không tìm thấy frames {'' if reprocess else 'mới '} nào cho video '{video_name}'")
        return True
    
    # Chọn samples tương ứng
    video_frames = frames.select(frame_ids)
    total_frames = len(frame_ids)
    
    print(f"Tìm thấy {total_frames} frames {'' if reprocess else 'mới '} cần xử lý phát hiện người cho video '{video_name}'")

    # Thiết lập tham số xử lý
    batch_size = 8  # Batch size nhỏ để tránh OOM
    confidence_threshold = 0.5

    # Thử sử dụng model zoo trước
    print("\nĐang tải mô hình phát hiện người...")
    try:
        # Sử dụng model zoo thay vì raw PyTorch model
        detector = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
        use_fiftyone_model = True
        print("Sử dụng FiftyOne Model Zoo để phát hiện người")
    except Exception as e:
        print(f"Không thể tải mô hình từ Model Zoo: {str(e)}")
        print("Chuyển sang sử dụng OpenCV DNN để phát hiện người")
        use_fiftyone_model = False

    print(f"\nĐang thực hiện nhận diện người cho video '{video_name}'...")

    # Lấy tất cả IDs của frames cần xử lý
    batch_ids_all = video_frames.values("id")
    
    # Xử lý theo từng nhóm nhỏ để tránh OOM
    people_count = 0
    batch_count = (len(batch_ids_all) + batch_size - 1) // batch_size
    
    for i in range(batch_count):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(batch_ids_all))
        
        if start_idx >= len(batch_ids_all):
            break
            
        # Lấy batch IDs từ danh sách đã tính trước
        current_batch_ids = batch_ids_all[start_idx:end_idx]
        
        if not current_batch_ids:
            continue
        
        # Lấy view chứa frames trong batch hiện tại
        batch_view = frames.select(current_batch_ids)
        batch_size_actual = len(batch_view)
        
        if batch_size_actual == 0:
            continue
            
        print(f"Xử lý batch {i+1}/{batch_count} ({start_idx}-{end_idx}/{len(batch_ids_all)})...")
        
        try:
            if use_fiftyone_model:
                # Áp dụng mô hình FiftyOne trên batch
                results = batch_view.apply_model(
                    detector,
                    label_field="person_detections",
                    confidence_thresh=confidence_threshold,
                    classes=["person"],
                )
                
                # Đếm số người
                for sample in batch_view:
                    if sample.has_field("person_detections") and len(sample.person_detections.detections) > 0:
                        people_count += len(sample.person_detections.detections)
            else:
                # Phát hiện người bằng OpenCV DNN
                for sample in batch_view:
                    try:
                        boxes, confidences = detect_people_with_opencv(sample.filepath, confidence_threshold)
                        
                        # Tạo detections cho mỗi người
                        if boxes:
                            from fiftyone.core.labels import Detections, Detection
                            detections = []
                            
                            for box, conf in zip(boxes, confidences):
                                detections.append(
                                    Detection(
                                        label="person",
                                        bounding_box=box,
                                        confidence=conf
                                    )
                                )
                            
                            sample["person_detections"] = Detections(detections=detections)
                            sample.save()
                            people_count += len(boxes)
                    except Exception as e:
                        print(f"Lỗi khi xử lý ảnh {sample.filepath}: {str(e)}")
            
            # Giải phóng bộ nhớ
            gc.collect()
            
            # Xóa dữ liệu không cần thiết
            del batch_view
            gc.collect()
            
        except Exception as e:
            print(f"Lỗi khi xử lý batch {i+1}: {e}")
            continue

    # Lấy tổng số frames có người chỉ trong video này
    # Đếm thủ công để tránh lỗi MongoDB
    frames_with_people = 0
    total_video_frames = 0
    
    print("Đang tính toán thống kê...")
    for sample in frames:
        if pattern in sample.filepath:
            total_video_frames += 1
            if sample.has_field("person_detections") and len(sample.person_detections.detections) > 0:
                frames_with_people += 1
    
    print(f"\nĐã hoàn thành nhận diện người cho video '{video_name}'")
    print(f"Số frames được xử lý: {total_frames}")
    print(f"Số người được nhận diện: {people_count}")
    print(f"Tổng số frames có người trong video này: {frames_with_people}/{total_video_frames} ({frames_with_people/total_video_frames*100:.2f}% nếu có frames)")
    print("Truy cập http://localhost:5151 để xem kết quả")
    
    return True

if __name__ == "__main__":
    # Tạo parser cho tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Detect people in frames of a specific video")
    parser.add_argument("video_name", help="Name of the video to process (e.g., 'test', 'test2')")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess all frames, even if they already have person_detections")
    
    # Parse tham số
    args = parser.parse_args()
    
    # Chạy hàm xử lý với video được chỉ định
    detect_people_for_video(args.video_name, args.reprocess) 