import numpy as np
import fiftyone as fo
import fiftyone.core.labels as fol
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
import json
import os
import gc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# === CONFIG ===
DATASET_NAME = "video_dataset_faces_deepface_arcface_retinaface_image_17/6"
EMBEDDING_FIELD = "face_embeddings"
OUTPUT_FIELD = "character_cluster"

def optimize_clustering_parameters(embeddings_array, min_samples_range=[3, 5, 7], 
                                 eps_range=np.arange(0.5, 1.2, 0.1)):
    """
    Tự động tối ưu parameters cho DBSCAN với ưu tiên ít cụm hơn nhưng có ý nghĩa
    
    Parameters:
    - embeddings_array: Mảng các embedding vectors
    - min_samples_range: List các giá trị min_samples cần thử (tăng để tránh cụm nhỏ)
    - eps_range: List các giá trị eps cần thử (tăng để merge cụm gần nhau)
    
    Returns:
    - best_params: Dict chứa các tham số tối ưu
    """
    print("Đang tối ưu hóa tham số clustering với ưu tiên ít cụm hơn...")
    
    best_score = -1
    best_params = {'min_samples': 3, 'eps': 0.4}
    best_labels = None
    best_n_clusters = 0
    best_n_noise = 0
    
    n_samples = len(embeddings_array)
    
    # Điều chỉnh tham số dựa trên kích thước dataset với cân bằng tốt hơn
    if n_samples < 50:
        min_samples_range = [3, 4, 5]
        # Bao gồm cả khoảng epsilon từ k-distance analysis
        eps_range = np.concatenate([
            np.arange(0.3, 0.5, 0.05),  # Epsilon nhỏ từ k-distance
            np.arange(0.5, 1.0, 0.1)    # Epsilon lớn để merge
        ])
    elif n_samples < 100:
        min_samples_range = [3, 5, 7]
        eps_range = np.concatenate([
            np.arange(0.25, 0.5, 0.05),
            np.arange(0.5, 1.1, 0.1)
        ])
    else:
        min_samples_range = [5, 7, 10]
        eps_range = np.concatenate([
            np.arange(0.2, 0.5, 0.05),
            np.arange(0.5, 1.2, 0.1)
        ])

    total_combinations = len(min_samples_range) * len(eps_range)
    current_combo = 0

    # If the array is too large, use a subset for optimization
    if len(embeddings_array) > 1000:
        print(f"Dataset quá lớn ({len(embeddings_array)} mẫu). Sử dụng tập con 1000 mẫu để tối ưu hóa.")
        indices = np.random.choice(len(embeddings_array), 1000, replace=False)
        opt_embeddings = embeddings_array[indices]
    else:
        opt_embeddings = embeddings_array

    results = []
    
    print(f"Thử các tham số với min_samples={min_samples_range}")
    print(f"Epsilon range: {eps_range.min():.2f} - {eps_range.max():.2f} ({len(eps_range)} values)")
    
    for min_samples in min_samples_range:
        for eps in eps_range:
            current_combo += 1
            if current_combo % 5 == 1:  # In ít hơn để tránh spam
                print(f"Progress: {current_combo}/{total_combinations} - Testing min_samples={min_samples}, eps={eps:.3f}")
            
            cluster = DBSCAN(min_samples=min_samples, eps=eps, metric='euclidean')
            labels = cluster.fit_predict(opt_embeddings)

            # Đánh giá kết quả clustering
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels != -1])  # Không tính nhiễu (-1)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels) * 100
            
            # Tính phân bố kích thước cụm
            cluster_sizes = []
            for label in unique_labels:
                if label != -1:
                    cluster_size = np.sum(labels == label)
                    cluster_sizes.append(cluster_size)
            
            # Đánh giá chất lượng clustering với scoring mới
            quality_score = 0
            
            # Ưu tiên số lượng cụm hợp lý (2-5 cụm)
            if 2 <= n_clusters <= 3:
                quality_score += 150  # Ưu tiên cao nhất cho 2-3 cụm
            elif 4 <= n_clusters <= 5:
                quality_score += 120
            elif 6 <= n_clusters <= 8:
                quality_score += 80
            elif n_clusters == 1:
                quality_score += 30  # Chấp nhận 1 cụm nhưng điểm thấp
            else:
                quality_score += 10
            
            # Ưu tiên tỉ lệ nhiễu hợp lý
            if noise_ratio < 15:
                quality_score += 50
            elif noise_ratio < 25:
                quality_score += 35
            elif noise_ratio < 40:
                quality_score += 20
            else:
                quality_score += 0
            
            # Ưu tiên phân bố cụm cân bằng
            if len(cluster_sizes) > 1:
                largest_cluster_ratio = max(cluster_sizes) / len(opt_embeddings) * 100
                smallest_cluster_ratio = min(cluster_sizes) / len(opt_embeddings) * 100
                
                # Tính coefficient of variation để đánh giá độ cân bằng
                if len(cluster_sizes) >= 2:
                    cv = np.std(cluster_sizes) / np.mean(cluster_sizes)
                    if cv < 0.5:  # Phân bố cân bằng
                        quality_score += 40
                    elif cv < 1.0:
                        quality_score += 25
                    else:
                        quality_score += 10
                
                # Tránh cụm quá nhỏ
                if smallest_cluster_ratio >= 5:  # Cụm nhỏ nhất >= 5% tổng
                    quality_score += 30
                elif smallest_cluster_ratio >= 2:
                    quality_score += 15
            
            # Tính silhouette score nếu có ít nhất 2 cụm
            silhouette = 0
            if n_clusters >= 2 and noise_ratio < 70:
                try:
                    # Loại bỏ điểm nhiễu khi tính silhouette
                    mask = labels != -1
                    if np.sum(mask) > n_clusters:
                        silhouette = silhouette_score(opt_embeddings[mask], labels[mask])
                        quality_score += silhouette * 100  # Tăng trọng số silhouette
                except Exception as e:
                    pass
            
            # In thông tin chi tiết cho các kết quả tốt
            if quality_score > 100 or n_clusters >= 2:
                print(f"  → min_samples={min_samples}, eps={eps:.3f}: {n_clusters} cụm, {noise_ratio:.1f}% nhiễu, score={quality_score:.1f}")
                if silhouette > 0:
                    print(f"    Silhouette: {silhouette:.3f}")
            
            results.append({
                'min_samples': min_samples,
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
                'silhouette': silhouette,
                'cluster_sizes': cluster_sizes,
                'quality_score': quality_score
            })
            
            # Cập nhật best parameters dựa trên quality_score
            if quality_score > best_score:
                best_score = quality_score
                best_params = {'min_samples': min_samples, 'eps': eps}
                best_labels = labels
                best_n_clusters = n_clusters
                best_n_noise = n_noise

    # Nếu tất cả cấu hình chỉ cho 1 cụm, thử epsilon nhỏ hơn nữa
    if best_n_clusters <= 1:
        print("\nTất cả cấu hình chỉ cho 1 cụm. Thử epsilon nhỏ hơn để tạo nhiều cụm...")
        small_eps_range = np.arange(0.1, 0.35, 0.02)
        small_min_samples = [2, 3]
        
        for min_samples in small_min_samples:
            for eps in small_eps_range:
                cluster = DBSCAN(min_samples=min_samples, eps=eps, metric='euclidean')
                labels = cluster.fit_predict(opt_embeddings)
                
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels[unique_labels != -1])
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels) * 100
                
                if n_clusters >= 2:
                    print(f"  Tìm thấy {n_clusters} cụm với min_samples={min_samples}, eps={eps:.3f}")
                    
                    # Tính quality score cho cấu hình này
                    quality_score = 100 if n_clusters <= 5 else 50
                    if noise_ratio < 30:
                        quality_score += 30
                    
                    if quality_score > best_score:
                        best_score = quality_score
                        best_params = {'min_samples': min_samples, 'eps': eps}
                        best_n_clusters = n_clusters
                        best_n_noise = n_noise
                        print(f"    ✓ Cập nhật best params: score={quality_score:.1f}")
                        break
            if best_n_clusters >= 2:
                break
    
    # Sort results by quality_score and display top 5
    results.sort(key=lambda x: x['quality_score'], reverse=True)
    print(f"\nTop 5 cấu hình tốt nhất:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. min_samples={result['min_samples']}, eps={result['eps']:.3f}")
        print(f"   → {result['n_clusters']} cụm, {result['n_noise']} nhiễu ({result['noise_ratio']:.1f}%)")
        print(f"   → Điểm chất lượng: {result['quality_score']:.1f}")
        if result['silhouette'] > 0:
            print(f"   → Silhouette: {result['silhouette']:.3f}")
    
    print(f"\nTham số được chọn: min_samples={best_params['min_samples']}, eps={best_params['eps']:.3f}")
    print(f"  → {best_n_clusters} cụm, {best_n_noise} điểm nhiễu ({best_n_noise/len(opt_embeddings)*100:.1f}%)")
    
    return best_params

def get_optimal_eps(embeddings_array, min_samples=5, visualize=False):
    """
    Phương pháp k-distance để tìm epsilon tối ưu cho DBSCAN
    
    Parameters:
    - embeddings_array: Mảng các embedding vectors
    - min_samples: Số mẫu tối thiểu trong cụm
    - visualize: Có hiển thị biểu đồ k-distance không
    
    Returns:
    - eps_candidates: Danh sách các giá trị epsilon tiềm năng
    """
    print("Đang tìm giá trị epsilon tối ưu cho DBSCAN...")
    
    # Sử dụng khoảng cách Euclidean
    nbrs = NearestNeighbors(n_neighbors=min(20, len(embeddings_array)-1), metric='euclidean')
        nbrs.fit(embeddings_array)
        distances, _ = nbrs.kneighbors(embeddings_array)
        
        # Sắp xếp khoảng cách đến láng giềng thứ k
    distances = np.sort(distances, axis=0)
    k_distances = distances[:, -1]
    
    # Tính các phân vị khác nhau để làm ứng viên epsilon
    # Thêm nhiều phân vị thấp hơn để bắt buộc nhiều cụm hơn
    percentiles = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70]
    eps_candidates = [np.percentile(k_distances, p) for p in percentiles]
    
    print("Các ứng viên epsilon (dựa trên phân vị):")
    for p, eps in zip(percentiles, eps_candidates):
        print(f"  Phân vị {p}%: eps = {eps:.4f}")
    
    # Tìm điểm gãy trên đồ thị k-distance
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(k_distances)), np.sort(k_distances))
        plt.xlabel('Điểm dữ liệu')
        plt.ylabel('K-distance')
        plt.title('K-distance Graph')
        plt.grid(True)
        plt.savefig("/fiftyone/data/k_distance_graph.png")
        print("Đã lưu biểu đồ k-distance tại: /fiftyone/data/k_distance_graph.png")
    
    return eps_candidates

def cluster_faces(face_embeddings):
    """
    Nhóm khuôn mặt sử dụng DBSCAN giống hệt như trong repo Face Clustering gốc:
    - metric='euclidean'
    - n_jobs=-1 (sử dụng tất cả CPU cores)
    - Không có tham số eps và min_samples tùy chỉnh (sử dụng mặc định)
    """
    if not face_embeddings:
        print("Không có embedding nào để nhóm.")
        return []

    encodings = np.array(face_embeddings)
    print(f"Shape của encoding array: {encodings.shape}")

    print("[INFO] clustering...")
    # Tạo DBSCAN object cho clustering encodings với metric "euclidean"
    # Giống hệt như trong repo gốc
    clt = DBSCAN(metric="euclidean", n_jobs=-1)
    clt.fit(encodings)
    
    # Lấy labels từ clustering result
    labels = clt.labels_
    
    # Xác định tổng số unique faces được tìm thấy trong dataset
    labelIDs = np.unique(labels)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))
    
    # In thông tin chi tiết về từng cluster
    for labelID in labelIDs:
        idxs = np.where(labels == labelID)[0]
        if labelID == -1:
            print("[INFO] outlier faces: {}".format(len(idxs)))
        else:
            print("[INFO] faces for face ID {}: {}".format(labelID, len(idxs)))
    
    return labels

def group_faces_into_characters():
    """Nhóm các khuôn mặt trong dataset thành các nhân vật theo từng video riêng biệt"""
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Load dataset khuôn mặt
    face_dataset_name = "video_dataset_faces_deepface_arcface_retinaface_image_17/6"
    if not fo.dataset_exists(face_dataset_name):
        print(f"Không tìm thấy dataset '{face_dataset_name}'")
        return
    
    face_dataset = fo.load_dataset(face_dataset_name)
    print(f"Đã load dataset '{face_dataset_name}' với {len(face_dataset)} frames")
    
    # Load video dataset để lấy thông tin videos
    video_dataset_name = "video_dataset"
    if not fo.dataset_exists(video_dataset_name):
        print(f"Không tìm thấy dataset '{video_dataset_name}'")
        return
    
    video_dataset = fo.load_dataset(video_dataset_name)
    print(f"Đã load dataset '{video_dataset_name}' với {len(video_dataset)} videos")
    
    # Tạo map video_name -> video_id để tổ chức dữ liệu
    videos_map = {}
    for video in video_dataset:
        video_name = os.path.basename(video.filepath)
        video_id = os.path.splitext(video_name)[0]  # Bỏ phần đuôi
        videos_map[video_id] = {
            "name": video_name,
            "id": video.id,
            "fps": video.metadata.frame_rate if hasattr(video.metadata, "frame_rate") else 24
        }
    
    # Phân loại frames theo video sử dụng sample_id hoặc video_id
    print("Phân loại frames theo video nguồn dựa trên sample_id...")
    frames_by_video = {}
    
    # Thu thập frames theo video
    batch_size = 50
    total_samples = len(face_dataset)
    batch_count = (total_samples + batch_size - 1) // batch_size
    
    for i in range(batch_count):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        
        print(f"Đang xử lý batch {i+1}/{batch_count} (samples {start_idx}-{end_idx}/{total_samples})...")
        
        # Lấy batch samples
        batch_ids = face_dataset.values("id")[start_idx:end_idx]
        batch_samples = face_dataset.select(batch_ids)
        
        for sample in batch_samples:
            if not sample.has_field("face_embeddings"):
                continue
            
            # Xác định video_id từ các trường dữ liệu
            video_id = None
            
            # Cách 1: Sử dụng trường video_id nếu có
            if sample.has_field("video_id"):
                video_id = sample.video_id
            # Cách 2: Từ sample_id nếu có định dạng phù hợp
            elif sample.has_field("sample_id"):
                # Giả sử sample_id có thể chứa thông tin video
                sample_id_parts = sample.sample_id.split('_')
                if len(sample_id_parts) > 0:
                    video_id = sample_id_parts[0]  # Lấy phần đầu tiên của sample_id
            # Cách 3: Phân tích filepath nếu các cách trên không khả dụng
            else:
                filepath = sample.filepath
                parts = filepath.split('/')
                # Tìm tên video từ đường dẫn (giả định cấu trúc: /path/to/frames/video_id/frame.jpg)
                for j, part in enumerate(parts):
                    if part == "frames" and j < len(parts) - 1:
                        video_id = parts[j+1]
                        break
            
            if video_id is None or video_id not in videos_map:
                # Nếu là dataset ảnh, gán video_id mặc định
                video_id = "image_dataset"
                        if video_id not in videos_map:
                            videos_map[video_id] = {
                        "name": "image_dataset",
                                "id": video_id,
                        "fps": 1
                            }
                
            # Thêm frame vào danh sách của video tương ứng
            if video_id not in frames_by_video:
                frames_by_video[video_id] = []
                
            frames_by_video[video_id].append(sample.id)
    
    print(f"Tìm thấy {len(frames_by_video)} videos có frames với face embeddings")
    
    # Lưu trữ kết quả phân nhóm nhân vật cho tất cả video
    all_character_segments = {}  # video_id -> character_id -> segments
    
    # Xử lý từng video riêng biệt - đảm bảo thuật toán DBSCAN áp dụng riêng cho từng video
    for video_id, frame_ids in frames_by_video.items():
        print(f"\n{'='*50}")
        print(f"Xử lý video: {videos_map[video_id]['name']} với {len(frame_ids)} frames")
        print(f"{'='*50}")
        
        # Tạo view chỉ chứa frames của video này
        video_frames_view = face_dataset.select(frame_ids)
        
        # Thu thập embeddings cho video này
        all_embeddings = []
        sample_to_embeddings = {}
        sample_to_bboxes = {}
        sample_to_info = {}
        
        print(f"Thu thập embeddings từ {len(frame_ids)} frames...")
        
        for sample in video_frames_view:
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
        
        print(f"Đã thu thập {len(all_embeddings)} embeddings từ {len(sample_to_embeddings)} frames trong video {video_id}")
        
        if not all_embeddings:
            print(f"Không có embedding nào để nhóm trong video {video_id}. Bỏ qua.")
            continue
        
        # Nhóm các khuôn mặt bằng DBSCAN
        print(f"\nBước 1: Phân cụm khuôn mặt trong video {video_id} bằng DBSCAN")
        print("-" * 50)
        
        # Phân cụm với DBSCAN - CẢI TIẾN: truyền số frames để tính mật độ
        labels = cluster_faces(all_embeddings)
        
        # Tạo map character_id -> color
        colors = list(mcolors.TABLEAU_COLORS.values())
        character_colors = {}
        for label in set(labels):
            color_idx = label % len(colors)
            character_colors[label] = colors[color_idx]
        
        # Gán nhãn cụm vào dataset và tạo character_detections 
        label_index = 0
        character_to_frames = {}  # Map character_id -> danh sách frames
        
        print(f"Cập nhật dataset với kết quả clustering cho video {video_id}...")
        
        # Video prefix cho nhãn nhân vật, đảm bảo duy nhất trên toàn hệ thống
        video_prefix = f"Video_{video_id}"
        
        for sample_id in frame_ids:
            if sample_id not in sample_to_embeddings:
                continue
                
            embeddings = sample_to_embeddings[sample_id]
            bboxes = sample_to_bboxes[sample_id]
            sample_labels = labels[label_index:label_index + len(embeddings)]
            label_index += len(embeddings)

            sample = face_dataset[sample_id]
            if sample.has_field("face_embeddings") and sample.face_embeddings is not None:
                face_embeddings = sample.face_embeddings
            else:
                face_embeddings = []

            detections = []

            for i, label in enumerate(sample_labels):
                # Gán video_id và character_id vào face_embeddings
                if i < len(face_embeddings):
                    face_embeddings[i]["video_id"] = video_id
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
                        video_id=video_id,
                        character_id=int(label),
                        color=color
                    )
                    detections.append(detection)

            # Gán cả hai trường vào sample
            sample["face_embeddings"] = face_embeddings
            sample["character_detections"] = fol.Detections(detections=detections)
            sample.save()
        
        # Tạo temporal segments cho mỗi nhân vật trong video này
        print(f"Tạo temporal segments cho các nhân vật trong video {video_id}...")
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
        
        # Lưu thông tin segments của video này vào kết quả tổng hợp
        all_character_segments[video_id] = {}
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
            
            all_character_segments[video_id][character_id] = serializable_segments
        
        # In thống kê về nhân vật trong video này
        print(f"\nThống kê nhân vật trong video {video_id}:")
        character_counts = {}
        for character_id, frames in character_to_frames.items():
            character_counts[character_id] = len(frames)
        
        # Sắp xếp theo số lượng frames giảm dần
        sorted_characters = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Tổng số nhân vật: {len(sorted_characters)}")
        for i, (character_id, count) in enumerate(sorted_characters, 1):
            segments = character_segments.get(character_id, [])
            print(f"{i}. Character {character_id}: {count} frames, {len(segments)} segments")
    
    # Lưu thông tin temporal detections vào file JSON
    output_dir = "/fiftyone/data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "character_segments.json")
    
    print(f"Lưu thông tin segments của tất cả videos vào {output_file}...")
    with open(output_file, "w") as f:
        json.dump(all_character_segments, f, indent=2)
    
    # Tạo file đúng cấu trúc cho character_search.py
    character_index_file = os.path.join(output_dir, "character_index.json")
    character_index = {}
    
    for video_id, characters in all_character_segments.items():
        # Lấy tên file video đầy đủ
        video_name = videos_map.get(video_id, {}).get("name", f"{video_id}.mp4")
        fps = videos_map.get(video_id, {}).get("fps", 24)
        
        character_index[video_name] = {}
        
        for character_id, segments in characters.items():
            character_name = f"{video_id}_Character_{character_id}"  # Thêm video_id vào tên character
            character_index[video_name][character_name] = [
                {
                    "start_frame": segment["start_frame"],
                    "end_frame": segment["end_frame"],
                    "start_time": frame_to_timestamp(segment["start_frame"], fps),
                    "end_time": frame_to_timestamp(segment["end_frame"], fps),
                    "duration": segment["end_frame"] - segment["start_frame"]
                }
                for segment in segments
            ]
    
    with open(character_index_file, "w") as f:
        json.dump(character_index, f, indent=2)
    
    print(f"Đã lưu character_index.json với {len(character_index)} videos")
    
    # In kết quả tổng hợp
    print("\nKết quả phân nhóm nhân vật theo từng video:")
    for video_id, characters in all_character_segments.items():
        print(f"{videos_map.get(video_id, {}).get('name', video_id)}: {len(characters)} nhân vật")
    
    print("\nTruy cập http://localhost:5151 để xem kết quả")

def frame_to_timestamp(frame_number, fps=24):
    """Convert frame number to timestamp string"""
    seconds = frame_number / fps
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def get_optimal_eps_elbow(embeddings_array, min_samples=5, max_eps=1.5, n_points=20):
    """
    Sử dụng phương pháp Elbow để tìm epsilon tối ưu cho DBSCAN
    
    Parameters:
    - embeddings_array: Mảng embedding vectors
    - min_samples: Tham số min_samples cho DBSCAN
    - max_eps: Giá trị epsilon tối đa để thử
    - n_points: Số điểm epsilon để thử
    
    Returns:
    - optimal_eps: Giá trị epsilon tối ưu
    - eps_candidates: List các giá trị epsilon đã thử
    - n_clusters_list: List số cụm tương ứng
    """
    print("Sử dụng phương pháp Elbow để tìm epsilon tối ưu cho DBSCAN...")
    
    # Tạo danh sách epsilon candidates
    eps_candidates = np.linspace(0.1, max_eps, n_points)
    n_clusters_list = []
    noise_ratios = []
    
    print(f"Thử {n_points} giá trị epsilon từ 0.1 đến {max_eps}")
    
    for eps in eps_candidates:
        # Chạy DBSCAN với epsilon hiện tại
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(embeddings_array)
        
        # Tính số cụm và tỉ lệ nhiễu
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels) * 100
        
        n_clusters_list.append(n_clusters)
        noise_ratios.append(noise_ratio)
        
        if eps % 0.2 < 0.1:  # In một số kết quả mẫu
            print(f"  eps={eps:.2f}: {n_clusters} cụm, {noise_ratio:.1f}% nhiễu")
    
    # Tìm điểm elbow trong đường cong số cụm
    # Elbow point là nơi tốc độ giảm số cụm chậm lại
    
    # Tính đạo hàm bậc nhất (rate of change)
    cluster_changes = np.diff(n_clusters_list)
    
    # Tính đạo hàm bậc hai (curvature)
    cluster_curvatures = np.diff(cluster_changes)
    
    # Tìm điểm có curvature lớn nhất (elbow point)
    # Nhưng cần đảm bảo không phải là những epsilon quá nhỏ (quá nhiều cụm)
    valid_indices = []
    for i, (n_clusters, noise_ratio) in enumerate(zip(n_clusters_list, noise_ratios)):
        # Chỉ xem xét những điểm có số cụm hợp lý và nhiễu không quá nhiều
        if 1 <= n_clusters <= 10 and noise_ratio < 80:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print("Không tìm thấy epsilon phù hợp, sử dụng giá trị mặc định")
        return 0.3, eps_candidates, n_clusters_list
    
    # Trong số các điểm hợp lệ, tìm điểm có số cụm trong khoảng lý tưởng
    best_idx = valid_indices[0]
    best_score = -1
    
    for idx in valid_indices:
        n_clusters = n_clusters_list[idx]
        noise_ratio = noise_ratios[idx]
        eps = eps_candidates[idx]
        
        # Scoring: ưu tiên 2-5 cụm, nhiễu ít
        score = 0
        if 2 <= n_clusters <= 4:
            score += 100
        elif n_clusters == 5:
            score += 80
        elif n_clusters == 1:
            score += 40
        else:
            score += 20
            
        # Bonus cho tỉ lệ nhiễu thấp
        if noise_ratio < 20:
            score += 50
        elif noise_ratio < 40:
            score += 30
        elif noise_ratio < 60:
            score += 15
            
        if score > best_score:
            best_score = score
            best_idx = idx
    
    optimal_eps = eps_candidates[best_idx]
    optimal_clusters = n_clusters_list[best_idx]
    optimal_noise = noise_ratios[best_idx]
    
    print(f"Elbow method chọn: eps={optimal_eps:.3f}")
    print(f"  → {optimal_clusters} cụm, {optimal_noise:.1f}% nhiễu")
    
    return optimal_eps, eps_candidates, n_clusters_list

def evaluate_clustering_with_silhouette(embeddings_array, labels):
    """
    Đánh giá chất lượng clustering bằng Silhouette score và các metrics khác
    
    Parameters:
    - embeddings_array: Mảng embedding vectors  
    - labels: Nhãn cụm
    
    Returns:
    - evaluation_results: Dict chứa các metrics đánh giá
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels) * 100
    
    evaluation = {
        'n_clusters': n_clusters,
        'n_noise': n_noise, 
        'noise_ratio': noise_ratio,
        'silhouette_score': 0,
        'cluster_balance': 0,
        'overall_quality': 0
    }
    
    # Tính Silhouette score nếu có ít nhất 2 cụm
    if n_clusters >= 2:
        try:
            # Loại bỏ noise khi tính silhouette
            mask = labels != -1
            if np.sum(mask) > n_clusters:
                silhouette = silhouette_score(embeddings_array[mask], labels[mask])
                evaluation['silhouette_score'] = silhouette
                print(f"Silhouette score: {silhouette:.4f}")
        except Exception as e:
            print(f"Không thể tính silhouette score: {e}")
    
    # Đánh giá độ cân bằng của các cụm
    if n_clusters > 0:
        cluster_sizes = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label != -1:
                cluster_size = np.sum(labels == label)
                cluster_sizes.append(cluster_size)
        
        if len(cluster_sizes) > 1:
            # Coefficient of variation để đánh giá độ cân bằng
            cv = np.std(cluster_sizes) / np.mean(cluster_sizes)
            balance_score = max(0, 1 - cv)  # CV thấp = cân bằng tốt
            evaluation['cluster_balance'] = balance_score
            print(f"Cluster balance (CV): {cv:.3f} → score: {balance_score:.3f}")
    
    # Tính overall quality score
    quality = 0
    
    # Điểm cho số cụm hợp lý
    if 2 <= n_clusters <= 4:
        quality += 40
    elif n_clusters == 5:
        quality += 30
    elif n_clusters == 1:
        quality += 15
    else:
        quality += 5
    
    # Điểm cho silhouette score
    quality += evaluation['silhouette_score'] * 30
    
    # Điểm cho cluster balance
    quality += evaluation['cluster_balance'] * 15
    
    # Điểm cho tỉ lệ nhiễu thấp
    if noise_ratio < 15:
        quality += 15
    elif noise_ratio < 30:
        quality += 10
    elif noise_ratio < 50:
        quality += 5
    
    evaluation['overall_quality'] = quality
    
    print(f"Overall quality score: {quality:.1f}")
    
    return evaluation

def fine_tune_dbscan_with_silhouette(embeddings_array, base_eps, base_min_samples):
    """
    Fine-tune tham số DBSCAN sử dụng Silhouette score
    
    Parameters:
    - embeddings_array: Mảng embedding vectors
    - base_eps: Epsilon cơ bản từ Elbow method
    - base_min_samples: Min_samples cơ bản
    
    Returns:
    - best_params: Dict chứa tham số tối ưu
    - best_labels: Nhãn cụm tốt nhất
    """
    print("Fine-tuning DBSCAN parameters với Silhouette score...")
    
    # Tạo grid search quanh base parameters
    eps_range = np.linspace(base_eps * 0.7, base_eps * 1.3, 5)
    min_samples_range = [max(2, base_min_samples - 1), base_min_samples, base_min_samples + 1]
    
    best_score = -1
    best_params = {'eps': base_eps, 'min_samples': base_min_samples}
    best_labels = None
    best_evaluation = None
    
    print(f"Grid search: eps [{eps_range.min():.3f} - {eps_range.max():.3f}], min_samples {min_samples_range}")
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            # Chạy DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = dbscan.fit_predict(embeddings_array)
            
            # Đánh giá bằng metrics tổng hợp
            evaluation = evaluate_clustering_with_silhouette(embeddings_array, labels)
            
            quality_score = evaluation['overall_quality']
            
            print(f"  eps={eps:.3f}, min_samples={min_samples}: quality={quality_score:.1f}")
            
            # Cập nhật best params nếu tốt hơn
            if quality_score > best_score:
                best_score = quality_score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_labels = labels.copy()
                best_evaluation = evaluation
                print(f"    ✓ New best: {best_params}")
    
    print(f"\nBest parameters after fine-tuning:")
    print(f"  eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}")
    print(f"  → {best_evaluation['n_clusters']} cụm, {best_evaluation['noise_ratio']:.1f}% nhiễu")
    print(f"  → Silhouette: {best_evaluation['silhouette_score']:.3f}")
    print(f"  → Overall quality: {best_evaluation['overall_quality']:.1f}")
    
    return best_params, best_labels

def estimate_characters_by_distance_analysis(embeddings_array, max_clusters=8):
    """
    Ước tính số nhân vật bằng phân tích khoảng cách trung bình giữa các cụm
    
    Parameters:
    - embeddings_array: Mảng embedding vectors đã chuẩn hóa
    - max_clusters: Số cụm tối đa để thử
    
    Returns:
    - estimated_characters: Số nhân vật ước tính
    - distance_scores: Dict chứa điểm số cho mỗi số cụm
    """
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial.distance import pdist, squareform
    
    print("Ước tính số nhân vật bằng phân tích khoảng cách...")
    
    n_samples = len(embeddings_array)
    if n_samples <= max_clusters:
        max_clusters = max(2, n_samples // 2)
    
    # Tính ma trận khoảng cách
    distances = pdist(embeddings_array, metric='euclidean')
    dist_matrix = squareform(distances)
    
    distance_scores = {}
    range_n_clusters = range(2, max_clusters + 1)
    
    for n_clusters in range_n_clusters:
        # Sử dụng Agglomerative Clustering để phân cụm
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(embeddings_array)
        
        # Tính khoảng cách trung bình TRONG các cụm (intra-cluster)
        intra_cluster_dists = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_points = np.where(mask)[0]
            
            if len(cluster_points) > 1:
                cluster_distances = []
                for i in range(len(cluster_points)):
                    for j in range(i + 1, len(cluster_points)):
                        cluster_distances.append(dist_matrix[cluster_points[i], cluster_points[j]])
                intra_cluster_dists.extend(cluster_distances)
        
        avg_intra_dist = np.mean(intra_cluster_dists) if intra_cluster_dists else 0
        
        # Tính khoảng cách trung bình GIỮA các cụm (inter-cluster)
        inter_cluster_dists = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                mask_i = labels == i
                mask_j = labels == j
                
                points_i = np.where(mask_i)[0]
                points_j = np.where(mask_j)[0]
                
                if len(points_i) > 0 and len(points_j) > 0:
                    cluster_dists = []
                    for pi in points_i:
                        for pj in points_j:
                            cluster_dists.append(dist_matrix[pi, pj])
                    inter_cluster_dists.append(np.mean(cluster_dists))
        
        avg_inter_dist = np.mean(inter_cluster_dists) if inter_cluster_dists else 0
        
        # Tính separation ratio (tỉ lệ phân tách)
        separation_ratio = avg_inter_dist / avg_intra_dist if avg_intra_dist > 0 else 0
        
        # Tính cluster compactness (độ chặt của cụm)
        compactness = 1 / (1 + avg_intra_dist) if avg_intra_dist > 0 else 1
        
        # Tính điểm số tổng hợp
        score = separation_ratio * compactness
        
        # Bonus cho số cụm hợp lý với video (2-4 cụm)
        if 2 <= n_clusters <= 4:
            score *= 1.5
        elif n_clusters == 5:
            score *= 1.2
        
        distance_scores[n_clusters] = score
        
        print(f"  {n_clusters} cụm: intra={avg_intra_dist:.3f}, inter={avg_inter_dist:.3f}, separation={separation_ratio:.3f}, score={score:.3f}")
    
    # Chọn số cụm có điểm score cao nhất
    best_n_clusters = max(distance_scores.keys(), key=lambda k: distance_scores[k])
    best_score = distance_scores[best_n_clusters]
    
    print(f"Ước tính số nhân vật dựa trên distance analysis: {best_n_clusters} (score: {best_score:.3f})")
    
    return best_n_clusters, distance_scores

def calculate_optimal_min_samples(embeddings_array, estimated_characters, num_frames=None):
    """
    Tính min_samples tối ưu dựa trên số nhân vật ước tính và phân bố dữ liệu
    
    Parameters:
    - embeddings_array: Mảng embedding vectors
    - estimated_characters: Số nhân vật ước tính
    - num_frames: Số frames (optional)
    
    Returns:
    - optimal_min_samples: Giá trị min_samples tối ưu
    """
    n_embeddings = len(embeddings_array)
    
    print(f"Tính min_samples cho {estimated_characters} nhân vật ước tính...")
    
    # Logic 1: Dựa trên số mẫu trung bình mỗi nhân vật
    avg_samples_per_character = n_embeddings / estimated_characters
    print(f"Trung bình mẫu/nhân vật: {avg_samples_per_character:.1f}")
    
    # Logic 2: min_samples nên là tỉ lệ nhỏ của mẫu mỗi nhân vật
    # Để đảm bảo không bỏ sót nhân vật có ít appearance
    if avg_samples_per_character >= 20:
        # Nhiều mẫu → có thể dùng min_samples cao hơn
        min_samples_ratio = 0.15  # 15% số mẫu trung bình
    elif avg_samples_per_character >= 10:
        # Trung bình → dùng tỉ lệ vừa phải
        min_samples_ratio = 0.20  # 20% số mẫu trung bình  
    else:
        # Ít mẫu → dùng tỉ lệ thấp hơn
        min_samples_ratio = 0.30  # 30% số mẫu trung bình
    
    min_samples_from_ratio = max(2, int(avg_samples_per_character * min_samples_ratio))
    
    # Logic 3: Dựa trên mật độ dữ liệu tổng quát
    if n_embeddings < 30:
        min_samples_base = 2
    elif n_embeddings < 60:
        min_samples_base = 3
    elif n_embeddings < 100:
        min_samples_base = 4
    else:
        min_samples_base = 5
    
    # Logic 4: Xem xét số frames nếu có
    min_samples_frame_based = min_samples_base
    if num_frames is not None and num_frames > 0:
        faces_per_frame = n_embeddings / num_frames
        if faces_per_frame > 3:
            # Mật độ cao → có thể tăng min_samples
            min_samples_frame_based = min_samples_base + 1
        elif faces_per_frame < 1.5:
            # Mật độ thấp → giảm min_samples
            min_samples_frame_based = max(2, min_samples_base - 1)
    
    # Kết hợp các logic
    candidates = [min_samples_from_ratio, min_samples_base, min_samples_frame_based]
    
    # Chọn giá trị median để tránh extreme
    optimal_min_samples = int(np.median(candidates))
    
    # Giới hạn trong khoảng hợp lý
    optimal_min_samples = max(2, min(optimal_min_samples, int(avg_samples_per_character * 0.4)))
    
    print(f"Min_samples candidates: ratio={min_samples_from_ratio}, base={min_samples_base}, frame={min_samples_frame_based}")
    print(f"Min_samples tối ưu: {optimal_min_samples}")
    
    return optimal_min_samples

def main():
    fo.config.database_uri = "mongodb://mongo:27017"
    if not fo.dataset_exists(DATASET_NAME):
        print(f"Không tìm thấy dataset '{DATASET_NAME}'")
        return
    dataset = fo.load_dataset(DATASET_NAME)
    print(f"Đã load dataset '{DATASET_NAME}' với {len(dataset)} samples")

    # Thu thập toàn bộ embedding và mapping sample
    print("[INFO] loading encodings...")
    all_embeddings = []
    sample_map = []  # (sample_id, index_in_sample)
    for sample in dataset:
        if hasattr(sample, EMBEDDING_FIELD) and getattr(sample, EMBEDDING_FIELD):
            for idx, fe in enumerate(getattr(sample, EMBEDDING_FIELD)):
                if "embedding" in fe:
                    all_embeddings.append(fe["embedding"])
                    sample_map.append((sample.id, idx))

    if not all_embeddings:
        print("Không có embedding nào để clustering.")
        return

    print(f"Tổng số embeddings: {len(all_embeddings)}")

    # Clustering sử dụng phương pháp giống hệt repo Face Clustering gốc
    labels = cluster_faces(all_embeddings)

    # Gán nhãn cluster vào sample
    print("[INFO] updating dataset with cluster labels...")
    for (sample_id, idx), label in zip(sample_map, labels):
        sample = dataset[sample_id]
        fe_list = getattr(sample, EMBEDDING_FIELD)
        if idx < len(fe_list):
            fe_list[idx][OUTPUT_FIELD] = int(label)
        sample[EMBEDDING_FIELD] = fe_list
        sample.save()
    
    print(f"Đã gán nhãn cluster vào trường '{OUTPUT_FIELD}' cho từng face embedding.")
    print("Clustering hoàn tất!")

if __name__ == "__main__":
    main()
