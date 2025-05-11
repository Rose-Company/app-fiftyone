import numpy as np
import fiftyone as fo
import fiftyone.core.labels as fol
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors
import json
import os

def estimate_optimal_clusters(embeddings, max_clusters=10):
    """
    Ước tính số lượng cụm tối ưu sử dụng nhiều phương pháp
    """
    embeddings_array = np.array(embeddings)
    embeddings_array = normalize(embeddings_array)
    
    if len(embeddings_array) <= max_clusters:
        print("Số lượng mẫu quá ít, giảm max_clusters...")
        max_clusters = max(2, len(embeddings_array) // 2)
    
    print(f"Đang đánh giá từ 2 đến {max_clusters} clusters...")
    
    # 1. Phương pháp Silhouette
    range_n_clusters = range(2, max_clusters + 1)
    silhouette_avg_list = []
    
    for n_clusters in range_n_clusters:
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings_array)
        labels = clustering.labels_
        silhouette_avg = silhouette_score(embeddings_array, labels)
        silhouette_avg_list.append(silhouette_avg)
        print(f"  {n_clusters} clusters: Silhouette = {silhouette_avg:.4f}")
    
    # Tìm điểm có silhouette cao nhất
    optimal_clusters_silhouette = range_n_clusters[np.argmax(silhouette_avg_list)]
    print(f"Số cụm tối ưu theo Silhouette: {optimal_clusters_silhouette}")
    
    # 2. Phương pháp Elbow dựa trên inertia
    inertia_list = []
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embeddings_array)
        inertia_list.append(kmeans.inertia_)
    
    # Tính đạo hàm của inertia để tìm điểm gãy
    inertia_diffs = np.diff(inertia_list)
    inertia_diffs2 = np.diff(inertia_diffs)
    optimal_clusters_elbow = range_n_clusters[1 + np.argmax(inertia_diffs2)]
    print(f"Số cụm tối ưu theo Elbow: {optimal_clusters_elbow}")
    
    # 3. Phương pháp phân tích khoảng cách trung bình giữa các cụm
    distances = pdist(embeddings_array)
    dist_matrix = squareform(distances)
    
    avg_dists = []
    for n_clusters in range_n_clusters:
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings_array)
        labels = clustering.labels_
        
        # Tính khoảng cách trung bình giữa các cụm
        inter_cluster_dists = []
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                mask_i = labels == i
                mask_j = labels == j
                if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                    points_i = np.where(mask_i)[0]
                    points_j = np.where(mask_j)[0]
                    
                    cluster_dists = []
                    for pi in points_i:
                        for pj in points_j:
                            cluster_dists.append(dist_matrix[pi, pj])
                    
                    inter_cluster_dists.append(np.mean(cluster_dists))
        
        avg_dists.append(np.mean(inter_cluster_dists) if inter_cluster_dists else 0)
    
    # Tìm điểm có khoảng cách giữa các cụm lớn nhất
    optimal_clusters_dist = range_n_clusters[np.argmax(avg_dists) if avg_dists else 0]
    print(f"Số cụm tối ưu theo khoảng cách giữa các cụm: {optimal_clusters_dist}")
    
    # Kết hợp các phương pháp (có thể điều chỉnh trọng số)
    candidates = [optimal_clusters_silhouette, optimal_clusters_elbow, optimal_clusters_dist]
    # Chọn số cụm xuất hiện nhiều nhất trong các phương pháp
    from collections import Counter
    optimal_n_clusters = Counter(candidates).most_common(1)[0][0]
    
    print(f"\nSố nhân vật tối ưu: {optimal_n_clusters}")
    return optimal_n_clusters

def cluster_faces_kmeans(face_embeddings, n_clusters):
    """Nhóm khuôn mặt bằng K-means"""
    if not face_embeddings:
        print("Không có embedding nào để nhóm.")
        return []

    embeddings_array = np.array(face_embeddings)
    print(f"Shape của embedding array: {embeddings_array.shape}")
    
    # Chuẩn hóa embeddings
    embeddings_array = normalize(embeddings_array)
    
    # Cluster sử dụng K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embeddings_array)
    
    labels = kmeans.labels_
    
    # In thông tin phân bố cụm
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Phân bố kích thước clusters:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples")
    
    return labels

def cluster_faces_hierarchical(face_embeddings, n_clusters):
    """Nhóm khuôn mặt bằng Agglomerative Hierarchical Clustering"""
    if not face_embeddings:
        print("Không có embedding nào để nhóm.")
        return []

    embeddings_array = np.array(face_embeddings)
    print(f"Shape của embedding array: {embeddings_array.shape}")
    
    # Chuẩn hóa embeddings
    embeddings_array = normalize(embeddings_array)
    
    # Cluster sử dụng AHC
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings_array)
    
    labels = clustering.labels_
    
    # In thông tin phân bố cụm
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Phân bố kích thước clusters:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples")
    
    return labels

def group_faces_into_characters():
    """Nhóm các khuôn mặt trong dataset thành các nhân vật"""
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Load dataset khuôn mặt
    face_dataset_name = "video_dataset_faces"
    if not fo.dataset_exists(face_dataset_name):
        print(f"Không tìm thấy dataset '{face_dataset_name}'")
        return
    
    face_dataset = fo.load_dataset(face_dataset_name)
    print(f"Đã load dataset '{face_dataset_name}' với {len(face_dataset)} frames")
    
    # Thu thập tất cả embedding khuôn mặt
    all_embeddings = []
    sample_to_embeddings = {}
    sample_to_bboxes = {}
    sample_to_info = {}  # Thêm thông tin về frame và timestamp

    print("Thu thập embeddings từ dataset...")
    for sample in face_dataset:
        if sample.has_field("face_embeddings"):
            face_embeddings = sample.face_embeddings
            embeddings = [fe["embedding"] for fe in face_embeddings]
            bboxes = [fe.get("bounding_box", [0, 0, 1, 1]) for fe in face_embeddings]

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
    
    print(f"Đã thu thập {len(all_embeddings)} embeddings từ {len(sample_to_embeddings)} frames")
    
    if not all_embeddings:
        print("Không có embedding nào để nhóm. Kiểm tra lại bước trích xuất khuôn mặt.")
        return
    
    # Tự động ước tính số nhân vật tối ưu
    print("\nBước 1: Ước tính số nhân vật trong video")
    print("-" * 50)
    n_clusters = estimate_optimal_clusters(all_embeddings)
    
    # Nhóm các khuôn mặt bằng hai phương pháp, so sánh kết quả
    print(f"\nBước 2: Nhóm các khuôn mặt với {n_clusters} nhân vật")
    print("-" * 50)
    
    print(f"\n1. Nhóm các khuôn mặt bằng K-means với {n_clusters} clusters:")
    labels_kmeans = cluster_faces_kmeans(all_embeddings, n_clusters)
    
    print(f"\n2. Nhóm các khuôn mặt bằng Hierarchical Clustering với {n_clusters} clusters:")
    labels_hierarchical = cluster_faces_hierarchical(all_embeddings, n_clusters)
    
    # Chọn phương pháp tốt hơn (mặc định dùng Hierarchical)
    print("\nSử dụng kết quả từ Hierarchical Clustering...")
    labels = labels_hierarchical
    
    # Tạo map character_id -> color
    colors = list(mcolors.TABLEAU_COLORS.values())
    character_colors = {}
    for label in set(labels):
        color_idx = label % len(colors)
        character_colors[label] = colors[color_idx]
    
    # Gán nhãn cụm vào dataset và tạo character_detections
    label_index = 0
    character_to_frames = {}  # Map character_id -> danh sách frames
    
    print("Cập nhật dataset với kết quả clustering...")
    for sample_id in sample_to_embeddings:
        embeddings = sample_to_embeddings[sample_id]
        bboxes = sample_to_bboxes[sample_id]
        sample_labels = labels[label_index:label_index + len(embeddings)]
        label_index += len(embeddings)

        sample = face_dataset[sample_id]
        if sample.has_field("face_embeddings"):
            face_embeddings = sample.face_embeddings
        else:
            face_embeddings = []

        detections = []

        for i, label in enumerate(sample_labels):
            # Gán character_id vào face_embeddings
            if i < len(face_embeddings):
                face_embeddings[i]["character_id"] = int(label)

            # Thêm frame vào map character_to_frames
            if label not in character_to_frames:
                character_to_frames[label] = []
            
            character_to_frames[label].append({
                "sample_id": sample.id,
                "frame_number": sample_to_info[sample.id]["frame_number"],
                "timestamp": sample_to_info[sample.id]["timestamp"],
                "filepath": sample_to_info[sample.id]["filepath"]
            })

            # Tạo detection để hiển thị rõ trên UI
            if i < len(bboxes):
                color = character_colors.get(label)
                detection = fol.Detection(
                    label=f"Character {label}",
                    bounding_box=bboxes[i],
                    confidence=1.0,
                    index=i,
                    character_id=int(label),
                    color=color
                )
                detections.append(detection)

        # Gán cả hai trường vào sample
        sample["face_embeddings"] = face_embeddings
        sample["character_detections"] = fol.Detections(detections=detections)
        sample.save()
    
    # Phần còn lại của code giữ nguyên
    # Tạo temporal segments cho mỗi nhân vật
    print("Tạo temporal segments cho các nhân vật...")
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
    output_file = os.path.join(output_dir, "character_segments.json")
    
    print(f"Lưu thông tin segments vào {output_file}...")
    with open(output_file, "w") as f:
        json_serializable = {}
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
            json_serializable[str(character_id)] = serializable_segments
        
        json.dump(json_serializable, f, indent=2)
    
    # Tạo view đã phân loại để dễ dàng sử dụng
    print("Tạo view đã phân loại...")
    character_view = face_dataset.view()
    
    # In thống kê về nhân vật
    print("\nThống kê về nhân vật:")
    character_counts = {}
    for character_id, frames in character_to_frames.items():
        character_counts[character_id] = len(frames)
    
    # Sắp xếp theo số lượng frames giảm dần
    sorted_characters = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Tổng số nhân vật: {len(sorted_characters)}")
    print("Top 10 nhân vật xuất hiện nhiều nhất:")
    for i, (character_id, count) in enumerate(sorted_characters[:10], 1):
        segments = character_segments.get(character_id, [])
        print(f"{i}. Character {character_id}: {count} frames, {len(segments)} segments")
    
    # Lưu view và in link để xem trên UI
    view_name = "characters_view_auto"
    character_view.save(view_name)
    print(f"\nĐã nhóm xong các khuôn mặt. Tổng số cụm: {len(set(labels))}")
    print("Truy cập http://localhost:5151 để xem kết quả")
    print(f"Chọn view '{view_name}' để xem các nhân vật đã được gán nhãn")

if __name__ == "__main__":
    print("Phân tích tự động số lượng nhân vật...")
    group_faces_into_characters()