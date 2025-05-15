import numpy as np
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors
import json
import os
import gc
import argparse
import datetime
import sys

# Import các hàm từ group_faces_characters.py
try:
    from group_faces_characters import estimate_optimal_clusters, cluster_faces_kmeans, cluster_faces_hierarchical, cluster_faces_dbscan, frame_to_timestamp
except ImportError:
    # Nếu không import được, sẽ thử import tương đối
    try:
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        sys.path.append(str(current_dir))
        from group_faces_characters import estimate_optimal_clusters, cluster_faces_kmeans, cluster_faces_hierarchical, cluster_faces_dbscan, frame_to_timestamp
    except ImportError:
        print("Lỗi: Không thể import các hàm từ group_faces_characters.py")
        print("Đảm bảo file group_faces_characters.py nằm trong cùng thư mục hoặc đường dẫn Python")
        sys.exit(1)

# Tự triển khai hàm cluster_faces_hdbscan thay vì import để đảm bảo tham số đúng
def cluster_faces_hdbscan(face_embeddings, min_cluster_size=5, min_samples=None):
    """Nhóm khuôn mặt bằng HDBSCAN"""
    try:
        import hdbscan
    except ImportError:
        print("Thư viện HDBSCAN không có sẵn. Cài đặt bằng: pip install hdbscan")
        return np.zeros(len(face_embeddings), dtype=int)
    
    if not face_embeddings:
        print("Không có embedding nào để nhóm.")
        return []

    embeddings_array = np.array(face_embeddings)
    print(f"Shape của embedding array: {embeddings_array.shape}")
    
    # Chuẩn hóa embeddings
    embeddings_array = normalize(embeddings_array)
    
    # Thiết lập min_samples nếu không được cung cấp
    if min_samples is None:
        min_samples = min_cluster_size // 2
    
    print(f"HDBSCAN với tham số: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    # Cluster sử dụng HDBSCAN với tham số chính xác như phiên bản cũ
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',  # Xác định rõ metric
        cluster_selection_epsilon=0.1,  # Giúp ghép các cụm nhỏ
        prediction_data=True  # Cần thiết để phân loại lại điểm nhiễu
    )
    
    labels = clusterer.fit_predict(embeddings_array)
    
    # Kiểm tra số lượng cụm và điểm nhiễu
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"HDBSCAN tìm thấy {n_clusters} cụm và {n_noise} điểm nhiễu ({n_noise/len(labels)*100:.1f}%)")
    
    # In thông tin phân bố cụm
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Phân bố kích thước clusters:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"Noise: {count} samples")
        else:
            print(f"Cluster {label}: {count} samples")
    
    # Xử lý điểm nhiễu sử dụng approximate_predict của HDBSCAN
    if -1 in labels and n_clusters > 0:
        print("\nĐang gán lại các điểm nhiễu sử dụng approximate_predict...")
        noise_indices = np.where(labels == -1)[0]
        
        # Gán mỗi điểm nhiễu vào cụm gần nhất sử dụng approximate_predict
        for idx in noise_indices:
            test_point = embeddings_array[idx].reshape(1, -1)
            cluster_label, _ = hdbscan.approximate_predict(clusterer, test_point)
            labels[idx] = cluster_label[0]
        
        # In thông tin sau khi gán lại
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Phân bố sau khi gán điểm nhiễu:")
        for label, count in zip(unique_labels, counts):
            if label == -1:
                print(f"Noise (vẫn còn): {count} samples")
            else:
                print(f"Cluster {label}: {count} samples")
                
        # Nếu vẫn còn điểm nhiễu, gán vào cụm gần nhất
        if -1 in labels:
            noise_indices = np.where(labels == -1)[0]
            valid_indices = np.where(labels != -1)[0]
            
            if len(valid_indices) > 0:  # Nếu có ít nhất một cụm hợp lệ
                # Tính trung bình của mỗi cụm
                cluster_centers = {}
                for label_id in set(labels):
                    if label_id != -1:
                        mask = labels == label_id
                        cluster_centers[label_id] = np.mean(embeddings_array[mask], axis=0)
                
                # Gán mỗi điểm nhiễu còn lại vào cụm gần nhất
                from scipy.spatial.distance import cdist
                for idx in noise_indices:
                    noise_point = embeddings_array[idx].reshape(1, -1)
                    distances = {
                        cluster_id: cdist(noise_point, center.reshape(1, -1), metric='euclidean')[0][0]
                        for cluster_id, center in cluster_centers.items()
                    }
                    closest_cluster = min(distances, key=distances.get)
                    labels[idx] = closest_cluster
            else:  # Nếu tất cả đều là nhiễu
                labels = np.zeros(len(labels), dtype=int)
    
    # Đảm bảo nhãn là các số nguyên liên tiếp từ 0
    if -1 in labels:
        # Nếu vẫn còn điểm nhiễu, gán vào một cụm mới
        labels[labels == -1] = n_clusters
        
    # Ánh xạ lại nhãn thành dãy số nguyên từ 0
    unique_labels = list(set(labels))
    mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    labels = np.array([mapping[label] for label in labels])
    
    # In thông tin sau khi ánh xạ
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Phân bố cuối cùng:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples")
    
    return labels

def group_faces_for_video(video_name, reprocess=False, create_new_dataset=False, force=False, method="hdbscan", no_backup=False):
    """
    Nhóm các khuôn mặt trong dataset thành các nhân vật cho một video cụ thể
    
    Args:
        video_name: Tên video (không có đuôi mở rộng, ví dụ: 'test', 'test2')
        reprocess: Nếu True, xử lý lại dù đã có character_detections
        create_new_dataset: Nếu True, tạo dataset mới thay vì ghi đè lên dataset cũ
        force: Nếu True, bỏ qua xác nhận khi xử lý lại dữ liệu
        method: Phương pháp phân cụm ("kmeans", "hierarchical", "dbscan", "hdbscan")
        no_backup: Nếu True, không tạo backup cho character_detections khi xử lý lại
    """
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Xác nhận khi xử lý lại dữ liệu
    if reprocess and not force:
        confirmation = input(f"CẢNH BÁO: Bạn sắp xử lý lại dữ liệu cho video '{video_name}'. Điều này có thể ghi đè lên character_detections đã có. Tiếp tục? (y/n): ")
        if confirmation.lower() != 'y':
            print("Đã hủy quá trình xử lý.")
            return False
    
    # Load dataset khuôn mặt
    face_dataset_name = "video_dataset_faces"
    if not fo.dataset_exists(face_dataset_name):
        print(f"Không tìm thấy dataset '{face_dataset_name}'")
        return False
    
    face_dataset = fo.load_dataset(face_dataset_name)
    print(f"Đã load dataset '{face_dataset_name}' với {len(face_dataset)} frames")
    
    if create_new_dataset:
        # Tạo tên dataset mới với timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dataset_name = f"video_dataset_characters_{video_name}_{timestamp}"
        new_dataset = fo.Dataset(new_dataset_name)
        print(f"Đã tạo dataset mới '{new_dataset_name}' để lưu kết quả")
    
    # Load video dataset để lấy thông tin videos
    video_dataset_name = "video_dataset"
    if not fo.dataset_exists(video_dataset_name):
        print(f"Không tìm thấy dataset '{video_dataset_name}'")
        return False
    
    video_dataset = fo.load_dataset(video_dataset_name)
    
    # Tìm thông tin video
    video_info = None
    for video in video_dataset:
        video_name_from_path = os.path.splitext(os.path.basename(video.filepath))[0]
        if video_name_from_path == video_name:
            video_info = {
                "name": os.path.basename(video.filepath),
                "id": video.id,
                "fps": video.metadata.frame_rate if hasattr(video.metadata, "frame_rate") else 24
            }
            break
    
    if not video_info:
        print(f"Không tìm thấy thông tin cho video '{video_name}' trong dataset.")
        # Tạo thông tin cơ bản nếu không tìm thấy
        video_info = {
            "name": f"{video_name}.mp4",
            "id": video_name,
            "fps": 24
        }
        print(f"Sử dụng thông tin mặc định: {video_info}")
    
    # Tìm các frames thuộc video này
    print(f"Đang tìm frames của video '{video_name}'...")
    
    # Pattern để lọc frames từ video này
    pattern = f"/frames/{video_name}/"
    
    # Tìm các frames có face_embeddings của video này
    frame_ids = []
    
    for sample in face_dataset:
        video_match = False
        
        # Cách 1: Kiểm tra theo trường video_id
        if sample.has_field("video_id") and sample.video_id == video_name:
            video_match = True
        # Cách 2: Kiểm tra theo filepath
        elif pattern in sample.filepath:
            video_match = True
        
        if video_match:
            # Kiểm tra nếu cần xử lý lại hoặc chưa có character_detections
            if reprocess or not sample.has_field("character_detections"):
                # Chỉ xử lý frames có face_embeddings
                if sample.has_field("face_embeddings") and sample.face_embeddings is not None and len(sample.face_embeddings) > 0:
                    frame_ids.append(sample.id)
    
    # Nếu không có frames nào cần xử lý thì kết thúc
    if not frame_ids:
        print(f"Không tìm thấy frames {'' if reprocess else 'mới '} cần xử lý nhóm khuôn mặt cho video '{video_name}'")
        return True
    
    # Tạo view chỉ chứa frames của video này
    video_frames = face_dataset.select(frame_ids)
    
    print(f"Tìm thấy {len(frame_ids)} frames {'' if reprocess else 'mới '} cần xử lý nhóm khuôn mặt cho video '{video_name}'")
    
    # Backup character_detections hiện có nếu đang xử lý lại
    if reprocess and not create_new_dataset and not no_backup:
        print("Đang lưu backup cho character_detections hiện có...")
        for sample in video_frames:
            if sample.has_field("character_detections"):
                sample["character_detections_backup"] = sample.character_detections
                
                # Backup các trường character_id trong face_embeddings
                if sample.has_field("face_embeddings") and sample.face_embeddings is not None:
                    for fe in sample.face_embeddings:
                        if "character_id" in fe:
                            fe["character_id_backup"] = fe["character_id"]
        
        # Lưu các thay đổi
        face_dataset.save()
        print("Đã lưu backup thành công.")
    
    # Thu thập embeddings cho video này
    print(f"\n{'='*50}")
    print(f"Đang thu thập embeddings từ {len(frame_ids)} frames...")
    
    all_embeddings = []
    sample_to_embeddings = {}
    sample_to_bboxes = {}
    sample_to_info = {}
    
    for sample in video_frames:
        if sample.has_field("face_embeddings") and sample.face_embeddings is not None:
            face_embeddings = sample.face_embeddings
            embeddings = [fe["embedding"] for fe in face_embeddings if "embedding" in fe]
            bboxes = [fe.get("bounding_box", [0, 0, 1, 1]) for fe in face_embeddings if "embedding" in fe]
            
            # Bỏ qua nếu không có embeddings hợp lệ
            if not embeddings or len(embeddings) == 0:
                continue

            all_embeddings.extend(embeddings)
            sample_to_embeddings[sample.id] = embeddings
            sample_to_bboxes[sample.id] = bboxes
            
            # Lưu thông tin frame và timestamp
            frame_number = 0
            timestamp = 0.0
            
            if sample.has_field("frame_number"):
                frame_number = sample.frame_number
            
            if sample.has_field("timestamp"):
                timestamp = sample.timestamp
                
            sample_to_info[sample.id] = {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "filepath": sample.filepath
            }
    
    print(f"Đã thu thập {len(all_embeddings)} embeddings từ {len(sample_to_embeddings)} frames trong video {video_name}")
    
    if not all_embeddings:
        print(f"Không có embedding nào để nhóm trong video {video_name}. Bỏ qua.")
        return True
    
    # Tự động ước tính số nhân vật tối ưu cho video này
    print(f"\nBước 1: Ước tính số nhân vật trong video {video_name}")
    print("-" * 50)
    n_clusters = estimate_optimal_clusters(all_embeddings)
    
    # Nhóm các khuôn mặt theo phương pháp được chọn
    print(f"\nBước 2: Nhóm các khuôn mặt trong video {video_name} bằng {method.upper()}")
    print("-" * 50)
    
    if method == "kmeans":
        labels = cluster_faces_kmeans(all_embeddings, n_clusters)
    elif method == "hierarchical":
        labels = cluster_faces_hierarchical(all_embeddings, n_clusters)
    elif method == "dbscan":
        labels = cluster_faces_dbscan(all_embeddings)
    else:  # hdbscan
        # Thay đổi tham số để phù hợp với phiên bản đã hoạt động tốt trước đây
        min_cluster_size = max(5, len(all_embeddings) // 30)  # Tăng ngưỡng chia so với 1/20
        
        # Nếu số lượng embeddings ít, giảm min_cluster_size để đảm bảo tìm được cụm
        if len(all_embeddings) < 100:
            min_cluster_size = max(3, len(all_embeddings) // 10)
            
        print(f"Sử dụng min_cluster_size = {min_cluster_size} cho {len(all_embeddings)} embeddings")
        
        try:
            # Thử với tham số giống code cũ
            labels = cluster_faces_hdbscan(all_embeddings, min_cluster_size=min_cluster_size)
            
            # Nếu không tìm thấy cụm nào, thử lại với tham số khác
            if len(set(labels)) <= 1:
                print("HDBSCAN không tìm thấy cụm. Thử với phương pháp dự phòng KMeans...")
                labels = cluster_faces_kmeans(all_embeddings, n_clusters)
        except Exception as e:
            print(f"Lỗi khi chạy HDBSCAN: {str(e)}. Sử dụng phương pháp KMeans thay thế...")
            labels = cluster_faces_kmeans(all_embeddings, n_clusters)
    
    # Tạo map character_id -> color
    colors = list(mcolors.TABLEAU_COLORS.values())
    character_colors = {}
    for label in set(labels):
        color_idx = label % len(colors)
        character_colors[label] = colors[color_idx]
    
    # Gán nhãn cụm vào dataset và tạo character_detections 
    label_index = 0
    character_to_frames = {}  # Map character_id -> danh sách frames
    
    print(f"Cập nhật dataset với kết quả clustering cho video {video_name}...")
    
    # Video prefix cho nhãn nhân vật, đảm bảo duy nhất trên toàn hệ thống
    video_prefix = f"Video_{video_name}"
    
    for sample_id in frame_ids:
        if sample_id not in sample_to_embeddings:
            continue
            
        embeddings = sample_to_embeddings[sample_id]
        bboxes = sample_to_bboxes[sample_id]
        sample_labels = labels[label_index:label_index + len(embeddings)]
        label_index += len(embeddings)

        # Lấy sample gốc hoặc tạo bản sao mới nếu dùng dataset mới
        if create_new_dataset:
            sample = face_dataset[sample_id].copy()
        else:
            sample = face_dataset[sample_id]
            
        if sample.has_field("face_embeddings") and sample.face_embeddings is not None:
            face_embeddings = sample.face_embeddings
        else:
            face_embeddings = []

        detections = []

        for i, label in enumerate(sample_labels):
            # Gán video_id và character_id vào face_embeddings
            if i < len(face_embeddings):
                face_embeddings[i]["video_id"] = video_name
                face_embeddings[i]["character_id"] = int(label)

            # Thêm frame vào map character_to_frames
            character_key = f"{label}"
            if character_key not in character_to_frames:
                character_to_frames[character_key] = []
            
            character_to_frames[character_key].append({
                "sample_id": sample.id,
                "frame_number": sample_to_info[sample.id]["frame_number"],
                "timestamp": sample_to_info[sample.id]["timestamp"],
                "filepath": sample_to_info[sample.id]["filepath"]
            })

            # Tạo detection để hiển thị rõ trên UI, với nhãn đặc trưng cho video
            if i < len(bboxes):
                color = character_colors.get(label)
                detection = fol.Detection(
                    label=f"{video_prefix}_Character_{label}",  # Thêm video_prefix vào nhãn
                    bounding_box=bboxes[i],
                    confidence=1.0,
                    index=i,
                    video_id=video_name,
                    character_id=int(label),
                    color=color
                )
                detections.append(detection)

        # Gán cả hai trường vào sample
        sample["face_embeddings"] = face_embeddings
        sample["character_detections"] = fol.Detections(detections=detections)
        
        # Lưu sample vào dataset thích hợp
        if create_new_dataset:
            new_dataset.add_sample(sample)
        else:
            sample.save()
    
    # Tạo temporal segments cho mỗi nhân vật trong video này
    print(f"Tạo temporal segments cho các nhân vật trong video {video_name}...")
    character_segments = {}
    
    for character_id, frames in character_to_frames.items():
        # Sắp xếp frames theo frame_number
        frames.sort(key=lambda x: x["frame_number"])
        
        segments = []
        if not frames:
            continue
            
        current_segment = {"start": frames[0], "end": frames[0], "frames": [frames[0]]}
        
        for i in range(1, len(frames)):
            current_frame = frames[i]
            prev_frame = frames[i-1]
            
            # Nếu frames liên tiếp (khoảng cách nhỏ), thêm vào segment hiện tại
            if current_frame["frame_number"] - prev_frame["frame_number"] <= 5:
                current_segment["end"] = current_frame
                current_segment["frames"].append(current_frame)
            else:
                # Lưu segment hiện tại và tạo mới
                segments.append(current_segment)
                current_segment = {"start": current_frame, "end": current_frame, "frames": [current_frame]}
        
        # Thêm segment cuối cùng
        segments.append(current_segment)
        character_segments[character_id] = segments
    
    # Lưu thông tin temporal detections vào file JSON
    output_dir = "/fiftyone/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo hoặc cập nhật file character_segments.json
    segments_file = os.path.join(output_dir, "character_segments.json")
    
    # Đọc file hiện có nếu tồn tại
    all_character_segments = {}
    if os.path.exists(segments_file):
        try:
            with open(segments_file, "r") as f:
                all_character_segments = json.load(f)
        except json.JSONDecodeError:
            print(f"Lỗi khi đọc file {segments_file}. Tạo file mới.")
    
    # Cập nhật thông tin segments cho video này
    all_character_segments[video_name] = {}
    for character_id, segments in character_segments.items():
        # Chuyển đổi thành cấu trúc có thể serialize
        serializable_segments = []
        for segment in segments:
            serializable_segment = {
                "start_frame": segment["start"]["frame_number"],
                "end_frame": segment["end"]["frame_number"],
                "start_time": segment["start"].get("timestamp", 0),
                "end_time": segment["end"].get("timestamp", 0),
                "frame_count": len(segment["frames"])
            }
            serializable_segments.append(serializable_segment)
        
        all_character_segments[video_name][character_id] = serializable_segments
    
    # Lưu file
    print(f"Lưu thông tin segments của video {video_name} vào {segments_file}...")
    with open(segments_file, "w") as f:
        json.dump(all_character_segments, f, indent=2)
    
    # Tạo hoặc cập nhật file character_index.json
    character_index_file = os.path.join(output_dir, "character_index.json")
    
    # Đọc file hiện có nếu tồn tại
    character_index = {}
    if os.path.exists(character_index_file):
        try:
            with open(character_index_file, "r") as f:
                character_index = json.load(f)
        except json.JSONDecodeError:
            print(f"Lỗi khi đọc file {character_index_file}. Tạo file mới.")
    
    # Cập nhật character_index cho video này
    fps = video_info.get("fps", 24)
    video_file_name = video_info.get("name", f"{video_name}.mp4")
    character_index[video_file_name] = {}
    
    for character_id, segments in character_segments.items():
        character_name = f"{video_name}_Character_{character_id}"  # Thêm video_id vào tên character
        character_index[video_file_name][character_name] = [
            {
                "start_frame": segment["start"]["frame_number"],
                "end_frame": segment["end"]["frame_number"],
                "start_time": frame_to_timestamp(segment["start"]["frame_number"], fps),
                "end_time": frame_to_timestamp(segment["end"]["frame_number"], fps),
                "duration": segment["end"]["frame_number"] - segment["start"]["frame_number"]
            }
            for segment in segments
        ]
    
    # Lưu file
    print(f"Lưu character_index.json với thông tin của video {video_name}...")
    with open(character_index_file, "w") as f:
        json.dump(character_index, f, indent=2)
    
    # In thống kê về nhân vật trong video này
    print(f"\nThống kê nhân vật trong video {video_name}:")
    character_counts = {}
    for character_id, frames in character_to_frames.items():
        character_counts[character_id] = len(frames)
    
    # Sắp xếp theo số lượng frames giảm dần
    sorted_characters = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Tổng số nhân vật: {len(sorted_characters)}")
    for i, (character_id, count) in enumerate(sorted_characters, 1):
        segments = character_segments.get(character_id, [])
        print(f"{i}. Character {character_id}: {count} frames, {len(segments)} segments")
    
    if create_new_dataset:
        print(f"\nĐã lưu kết quả vào dataset mới: '{new_dataset_name}'")
        print(f"Số lượng samples trong dataset mới: {len(new_dataset)}")
    
    print("\nTruy cập http://localhost:5151 để xem kết quả")
    return True

if __name__ == "__main__":
    # Tạo parser cho tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Nhóm khuôn mặt thành nhân vật cho một video cụ thể")
    parser.add_argument("video_name", help="Tên video để xử lý (ví dụ: 'test', 'test2')")
    parser.add_argument("--reprocess", action="store_true", help="Xử lý lại tất cả frames, ngay cả khi đã có character_detections")
    parser.add_argument("--new-dataset", action="store_true", help="Tạo dataset mới thay vì ghi đè lên dataset hiện có")
    parser.add_argument("--force", action="store_true", help="Bỏ qua xác nhận khi xử lý lại dữ liệu")
    parser.add_argument("--method", choices=["kmeans", "hierarchical", "dbscan", "hdbscan"], 
                        default="hdbscan", help="Phương pháp phân cụm (mặc định: hdbscan)")
    parser.add_argument("--no-backup", action="store_true", help="Không tạo backup cho character_detections khi xử lý lại")
    
    # Parse tham số
    args = parser.parse_args()
    
    # Ghi đè hàm input nếu force=True
    if args.force:
        original_input = input
        
        def force_yes(prompt):
            print(prompt + " y (forced)")
            return "y"
        
        input = force_yes
    
    try:
        group_faces_for_video(args.video_name, args.reprocess, args.new_dataset, args.force, args.method, args.no_backup)
    finally:
        # Khôi phục input gốc nếu đã ghi đè
        if args.force:
            input = original_input 