#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_embedding_analysis.py - Phân tích embeddings khuôn mặt dùng FiftyOne Brain

Script này thực hiện phân tích embeddings khuôn mặt đã được trích xuất bởi 
face_recognition_utils.py, sử dụng các phương pháp của FiftyOne Brain để
trực quan hóa, tìm kiếm khuôn mặt tương tự và đánh giá tính duy nhất.

Tác giả: Nguyễn Thanh Phát
"""

import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import ViewField as F
import numpy as np
import argparse
import datetime
import time
import os
import json
from tqdm import tqdm

def setup_connection():
    """Thiết lập kết nối đến MongoDB"""
    print("Thiết lập kết nối đến MongoDB...")
    fo.config.database_uri = "mongodb://mongo:27017"
    print("Kết nối thành công!")

def load_dataset(dataset_name="video_dataset_faces_dlib"):
    """Load dataset khuôn mặt"""
    if not fo.dataset_exists(dataset_name):
        print(f"Không tìm thấy dataset '{dataset_name}'")
        available_datasets = fo.list_datasets()
        if available_datasets:
            print(f"Các dataset có sẵn: {', '.join(available_datasets)}")
            alt_name = input(f"Nhập tên dataset khuôn mặt muốn sử dụng (hoặc Enter để kết thúc): ")
            if alt_name and fo.dataset_exists(alt_name):
                dataset_name = alt_name
            else:
                return None
        else:
            print("Không có dataset nào. Vui lòng chạy face_recognition_utils.py trước.")
            return None
    
    dataset = fo.load_dataset(dataset_name)
    print(f"Đã load dataset '{dataset_name}' với {len(dataset)} samples")
    return dataset

def get_video_frames(dataset, video_name):
    """Lấy frames của một video cụ thể"""
    # Pattern để lọc frames từ video này
    pattern = f"/frames/{video_name}/"
    
    # Sửa: Sử dụng re_match thay vì contains để tương thích với MongoDB
    view = dataset.match(F("filepath").re_match(f".*{pattern}.*"))
    
    # Hoặc cách 2: Lọc thủ công nếu re_match vẫn có vấn đề
    if len(view) == 0:
        frames = []
        for sample in dataset:
            if pattern in sample.filepath:
                frames.append(sample.id)
        view = dataset.select(frames) if frames else dataset.limit(0)
    
    print(f"Tìm thấy {len(view)} frames cho video '{video_name}'")
    return view

def collect_embeddings(view, embedding_field="face_embeddings", min_confidence=0.0):
    """Thu thập embeddings từ các frames"""
    print(f"Đang thu thập {embedding_field} từ {len(view)} frames...")
    
    all_embeddings = []
    all_sample_ids = []
    all_frame_numbers = []
    all_bbox_info = []
    sample_to_face_map = {}  # Map sample_id -> danh sách vị trí embeddings
    
    for sample in tqdm(view):
        if sample.has_field(embedding_field) and getattr(sample, embedding_field):
            sample_faces = []
            
            for i, fe in enumerate(getattr(sample, embedding_field)):
                if "embedding" in fe:
                    confidence = fe.get("confidence", 1.0)
                    if confidence >= min_confidence:
                        all_embeddings.append(np.array(fe["embedding"]))
                        all_sample_ids.append(sample.id)
                        
                        # Lấy frame_number nếu có
                        frame_number = sample.frame_number if sample.has_field("frame_number") else 0
                        all_frame_numbers.append(frame_number)
                        
                        # Lấy thông tin bounding box
                        bbox = fe.get("bounding_box", fe.get("bbox", [0, 0, 1, 1]))
                        all_bbox_info.append(bbox)
                        
                        # Thêm vị trí embedding vào map
                        sample_faces.append(len(all_embeddings) - 1)
            
            if sample_faces:
                sample_to_face_map[sample.id] = sample_faces
    
    result = {
        "embeddings": all_embeddings,
        "sample_ids": all_sample_ids,
        "frame_numbers": all_frame_numbers,
        "bbox_info": all_bbox_info,
        "sample_to_face_map": sample_to_face_map
    }
    
    print(f"Đã thu thập {len(all_embeddings)} embeddings từ {len(sample_to_face_map)} frames")
    return result

def compute_similarity(view, embedding_data, target_id=None, brain_key=None):
    """Tính toán similarity giữa các embedding"""
    print("\n=== Tính toán Similarity ===")
    
    if not embedding_data["embeddings"]:
        print("Không có embeddings để tính toán similarity")
        return None
    
    if brain_key is None:
        brain_key = f"face_similarity_{int(time.time())}"
    
    # Thay vì sử dụng compute_similarity của FiftyOne Brain, 
    # chúng ta sẽ tự tính toán ma trận khoảng cách
    from sklearn.metrics.pairwise import cosine_distances
    embeddings_array = np.array(embedding_data["embeddings"])
    
    # Tính ma trận khoảng cách cosine
    print(f"Đang tính toán ma trận khoảng cách cho {len(embeddings_array)} embeddings...")
    distance_matrix = cosine_distances(embeddings_array)
    
    # Lưu kết quả vào các sample
    print("Đang lưu kết quả tương tự vào samples...")
    
    # Tạo mapping từ embedding index sang (sample_id, embedding_index_in_sample)
    embedding_mapping = {}
    for i, sample_id in enumerate(embedding_data["sample_ids"]):
        if sample_id not in embedding_mapping:
            embedding_mapping[sample_id] = []
        embedding_mapping[sample_id].append(i)
    
    # Nếu không chỉ định target_id, chọn sample đầu tiên
    if target_id is None and embedding_data["sample_ids"]:
        target_id = embedding_data["sample_ids"][0]
    
    # Lưu điểm tương tự trung bình vào mỗi sample
    for sample_id, indices in embedding_mapping.items():
        sample = view[sample_id]
        
        # Tính điểm tương tự trung bình từ tất cả target embedding đến embedding của sample này
        if target_id in embedding_mapping:
            target_indices = embedding_mapping[target_id]
            similarities = []
            
            for target_idx in target_indices:
                for sample_idx in indices:
                    # Khoảng cách cosine -> similarity (1 - khoảng cách)
                    sim = 1.0 - distance_matrix[target_idx, sample_idx]
                    similarities.append(sim)
            
            # Gán điểm tương tự trung bình
            avg_similarity = np.mean(similarities) if similarities else 0.0
            sample[f"similarity_{target_id}"] = avg_similarity
            
        # Lưu thông tin chi tiết về các khuôn mặt trong sample
        face_similarities = []
        for idx in indices:
            # Tìm k khuôn mặt giống nhất
            k = min(10, len(embedding_data["embeddings"]))
            most_similar = np.argsort(distance_matrix[idx])[:k]
            
            similarities = []
            for sim_idx in most_similar:
                if sim_idx == idx:  # Bỏ qua chính nó
                    continue
                    
                sim_sample_id = embedding_data["sample_ids"][sim_idx]
                sim_frame = embedding_data["frame_numbers"][sim_idx]
                
                similarities.append({
                    "sample_id": sim_sample_id,
                    "frame_number": sim_frame,
                    "similarity": 1.0 - distance_matrix[idx, sim_idx]
                })
            
            face_similarities.append(similarities)
        
        # Thêm thông tin này vào sample
        if face_similarities:
            sample[f"{brain_key}_details"] = face_similarities
        
        # Lưu sample
        sample.save()
    
    print(f"Đã tính toán và lưu similarity với brain_key: {brain_key}")
    
    # Sửa: Thay đổi cách lấy top samples
    similar_view = view.sort_by(f"similarity_{target_id}", reverse=True).limit(5)
    
    print(f"\nTop 5 frames tương tự nhất với sample_id={target_id}:")
    
    # Duyệt qua các samples trực tiếp thay vì sử dụng values()
    for i, sample in enumerate(similar_view, 1):
        similarity = sample[f"similarity_{target_id}"]
        print(f"{i}. {os.path.basename(sample.filepath)}: {similarity:.4f}")
    
    return {"brain_key": brain_key, "target_id": target_id}

def compute_uniqueness(view, embedding_data, brain_key=None):
    """Tính toán uniqueness cho các embedding"""
    print("\n=== Tính toán Uniqueness ===")
    
    if not embedding_data["embeddings"]:
        print("Không có embeddings để tính toán uniqueness")
        return None
    
    if brain_key is None:
        brain_key = f"face_uniqueness_{int(time.time())}"
    
    # Tự tính toán uniqueness thay vì sử dụng FiftyOne Brain
    from sklearn.metrics.pairwise import cosine_distances
    import numpy as np
    
    embeddings_array = np.array(embedding_data["embeddings"])
    
    # Tính ma trận khoảng cách cosine 
    print(f"Đang tính toán ma trận khoảng cách cho {len(embeddings_array)} embeddings...")
    distance_matrix = cosine_distances(embeddings_array)
    
    # Tạo mapping từ embedding index sang sample_id
    embedding_mapping = {}
    for i, sample_id in enumerate(embedding_data["sample_ids"]):
        if sample_id not in embedding_mapping:
            embedding_mapping[sample_id] = []
        embedding_mapping[sample_id].append(i)
    
    # Tính uniqueness cho mỗi embedding
    print("Đang tính toán uniqueness...")
    
    # Khoảng cách trung bình từ mỗi embedding đến K embedding gần nhất
    k = min(10, len(embeddings_array) - 1)
    uniqueness_scores = np.zeros(len(embeddings_array))
    
    for i in range(len(embeddings_array)):
        # Lấy khoảng cách đến tất cả embeddings khác
        distances = distance_matrix[i]
        # Sắp xếp và lấy k khoảng cách nhỏ nhất (loại bỏ khoảng cách đến chính nó = 0)
        sorted_distances = np.sort(distances)[1:k+1]
        # Uniqueness là khoảng cách trung bình (càng xa càng duy nhất)
        uniqueness_scores[i] = np.mean(sorted_distances)
    
    # Tính uniqueness cho mỗi sample dựa trên trung bình của các embeddings trong đó
    sample_uniqueness = {}
    
    for sample_id, indices in embedding_mapping.items():
        # Tính điểm uniqueness trung bình cho sample này
        avg_uniqueness = np.mean([uniqueness_scores[idx] for idx in indices])
        sample_uniqueness[sample_id] = avg_uniqueness
        
        # Cập nhật sample trong dataset
        sample = view[sample_id]
        sample["uniqueness"] = avg_uniqueness
        
        # Lưu thông tin chi tiết về uniqueness của từng khuôn mặt
        face_uniqueness = []
        for idx in indices:
            face_uniqueness.append({
                "embedding_idx": idx,
                "uniqueness": float(uniqueness_scores[idx])
            })
        
        sample[f"{brain_key}_details"] = face_uniqueness
        sample.save()
    
    print(f"Đã tính toán và lưu uniqueness với brain_key: {brain_key}")
    
    # Hiển thị top samples có uniqueness cao nhất
    unique_view = view.sort_by("uniqueness", reverse=True).limit(5)
    
    print("\nTop 5 frames có độ duy nhất cao nhất:")
    for i, sample in enumerate(unique_view, 1):
        print(f"{i}. {os.path.basename(sample.filepath)}: {sample.uniqueness:.4f}")
    
    return {"brain_key": brain_key}

def compute_visualization(view, embedding_data, method="umap", brain_key=None, label_field=None):
    """Trực quan hóa embeddings"""
    print(f"\n=== Tính toán Visualization ({method.upper()}) ===")
    
    if not embedding_data["embeddings"]:
        print("Không có embeddings để trực quan hóa")
        return None
    
    if brain_key is None:
        brain_key = f"face_viz_{method}_{int(time.time())}"
    
    # Tạo nhãn cho visualization nếu không có
    if label_field is None:
        label_field = f"face_cluster_{method}"
        
        # Tạo nhãn đơn giản dựa trên frame_number
        frames_grouped = {}
        for idx, frame_num in enumerate(embedding_data["frame_numbers"]):
            frame_group = frame_num // 10  # Nhóm theo 10 frames
            if frame_group not in frames_grouped:
                frames_grouped[frame_group] = []
            frames_grouped[frame_group].append(idx)
        
        # Gán nhãn cho từng sample
        for sample_idx, sample_id in enumerate(embedding_data["sample_ids"]):
            frame_num = embedding_data["frame_numbers"][sample_idx]
            frame_group = frame_num // 10
            sample = view[sample_id]
            sample[label_field] = f"Group {frame_group}"
            sample.save()
        
        print(f"Đã tạo trường {label_field} cho visualization")
    
    # Thay vì sử dụng FiftyOne Brain, tự triển khai visualization
    embeddings_array = np.array(embedding_data["embeddings"])
    print(f"Đang tính toán visualization với phương pháp {method} cho {len(embeddings_array)} embeddings...")
    
    # Giảm chiều xuống còn 3 thành phần
    reduced_data = None
    
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=3, random_state=42)
            reduced_data = reducer.fit_transform(embeddings_array)
            print("Đã thực hiện giảm chiều với UMAP")
        except ImportError:
            print("Lỗi: Thư viện UMAP không có sẵn. Vui lòng cài đặt bằng lệnh:")
            print('docker exec -it fiftyone-docker-fiftyone-1 pip install "umap-learn>=0.5"')
            print("Hoặc thử lại với phương pháp khác (ví dụ: --viz-method tsne)")
            return None
    
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=3, random_state=42)
            reduced_data = reducer.fit_transform(embeddings_array)
            print("Đã thực hiện giảm chiều với t-SNE")
        except:
            print("Lỗi khi sử dụng t-SNE, chuyển sang PCA...")
            method = "pca"
    
    if method == "pca" or reduced_data is None:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3)
        reduced_data = reducer.fit_transform(embeddings_array)
        print("Đã thực hiện giảm chiều với PCA")
    
    # Tạo mapping từ embedding index sang sample_id
    embedding_mapping = {}
    for i, sample_id in enumerate(embedding_data["sample_ids"]):
        if sample_id not in embedding_mapping:
            embedding_mapping[sample_id] = []
        embedding_mapping[sample_id].append(i)
    
    # Lưu kết quả visualization vào các sample
    print("Đang lưu kết quả visualization vào samples...")
    
    # Trường để lưu tọa độ của từng embedding
    dim_fields = [f"{brain_key}_dim0", f"{brain_key}_dim1", f"{brain_key}_dim2"]
    
    for sample_id, indices in embedding_mapping.items():
        sample = view[sample_id]
        
        # Tính trung bình các tọa độ cho sample này
        sample_points = [reduced_data[idx] for idx in indices]
        avg_point = np.mean(sample_points, axis=0)
        
        # Lưu tọa độ trung bình
        for dim in range(3):
            sample[dim_fields[dim]] = float(avg_point[dim])
        
        # Lưu chi tiết về các embeddings trong sample
        embedding_coords = []
        for idx in indices:
            embedding_coords.append({
                "idx": idx,
                "coords": [float(reduced_data[idx][d]) for d in range(3)]
            })
        
        sample[f"{brain_key}_details"] = embedding_coords
        sample.save()
    
    print(f"Đã tính toán và lưu visualization với brain_key: {brain_key}")
    print(f"Tọa độ đã được lưu vào các trường: {', '.join(dim_fields)}")
    print("\nĐể xem visualization trong FiftyOne App:")
    print(f"1. Mở Plots Panel")
    print(f"2. Tạo một Scatterplot với:")
    print(f"   - X-axis: {dim_fields[0]}")
    print(f"   - Y-axis: {dim_fields[1]}")
    print(f"   - Color by: {label_field}")
    
    return {"brain_key": brain_key, "dim_fields": dim_fields, "label_field": label_field}

def find_similar_faces(view, embedding_data, query_embedding, k=10):
    """Tìm k khuôn mặt giống nhất với embedding truy vấn"""
    if not embedding_data["embeddings"]:
        print("Không có embeddings để tìm kiếm")
        return []
    
    # Tính khoảng cách cosine giữa query và tất cả embeddings
    from sklearn.metrics.pairwise import cosine_distances
    query = np.array(query_embedding).reshape(1, -1)
    all_embeddings = np.array(embedding_data["embeddings"])
    
    distances = cosine_distances(query, all_embeddings)[0]
    
    # Sắp xếp theo khoảng cách tăng dần
    similar_indices = np.argsort(distances)[:k]
    
    similar_faces = []
    for idx in similar_indices:
        sample_id = embedding_data["sample_ids"][idx]
        sample = view[sample_id]
        
        similar_faces.append({
            "sample_id": sample_id,
            "filepath": sample.filepath,
            "frame_number": embedding_data["frame_numbers"][idx],
            "distance": distances[idx],
            "bbox": embedding_data["bbox_info"][idx]
        })
    
    return similar_faces

def cluster_faces(embedding_data, method="kmeans", n_clusters=None):
    """Phân cụm các khuôn mặt"""
    print(f"\n=== Phân cụm khuôn mặt với {method} ===")
    
    if not embedding_data["embeddings"]:
        print("Không có embeddings để phân cụm")
        return None
    
    # Chuẩn hóa embeddings
    from sklearn.preprocessing import normalize
    embeddings_array = np.array(embedding_data["embeddings"])
    embeddings_array = normalize(embeddings_array)
    
    # Ước tính số cụm tối ưu nếu không chỉ định
    if n_clusters is None:
        if method == "kmeans":
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Sử dụng Silhouette score để ước tính số cụm tối ưu
            sil_scores = []
            max_clusters = min(20, len(embeddings_array) // 5)
            range_n_clusters = range(2, max_clusters + 1)
            
            print(f"Đang ước tính số cụm tối ưu từ 2 đến {max_clusters}...")
            for n in range_n_clusters:
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings_array)
                
                if len(set(labels)) <= 1:
                    continue
                    
                sil_score = silhouette_score(embeddings_array, labels)
                sil_scores.append(sil_score)
                print(f"  {n} clusters: Silhouette score = {sil_score:.4f}")
            
            if sil_scores:
                n_clusters = range_n_clusters[np.argmax(sil_scores)]
                print(f"Số cụm tối ưu: {n_clusters}")
            else:
                n_clusters = 3
                print(f"Không thể ước tính số cụm tối ưu. Sử dụng mặc định: {n_clusters}")
        else:
            n_clusters = 3
            print(f"Sử dụng số cụm mặc định: {n_clusters}")
    
    # Thực hiện phân cụm
    labels = None
    if method == "kmeans":
        from sklearn.cluster import KMeans
        print(f"Phân cụm với KMeans, {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        print(f"Phân cụm với Agglomerative Clustering, {n_clusters} clusters...")
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(embeddings_array)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        
        # Tìm epsilon tối ưu
        nbrs = NearestNeighbors(n_neighbors=min(20, len(embeddings_array)-1))
        nbrs.fit(embeddings_array)
        distances, _ = nbrs.kneighbors(embeddings_array)
        
        # Sắp xếp khoảng cách đến láng giềng thứ k
        k_distances = np.sort(distances[:, -1])
        
        # Chọn epsilon ở điểm gãy (có thể điều chỉnh phần trăm)
        epsilon = np.percentile(k_distances, 20)
        min_samples = 3
        
        print(f"Phân cụm với DBSCAN, epsilon={epsilon:.4f}, min_samples={min_samples}...")
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings_array)
    elif method == "hdbscan":
        try:
            import hdbscan
            min_cluster_size = max(5, len(embeddings_array) // 30)
            min_samples = min_cluster_size // 2
            
            print(f"Phân cụm với HDBSCAN, min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                prediction_data=True
            )
            labels = clusterer.fit_predict(embeddings_array)
            
            # Xử lý điểm nhiễu nếu có
            if -1 in labels:
                noise_count = np.sum(labels == -1)
                print(f"Tìm thấy {noise_count} điểm nhiễu ({noise_count/len(labels)*100:.1f}%)")
                
                # Gán điểm nhiễu vào cụm gần nhất
                if np.sum(labels != -1) > 0:  # Nếu có ít nhất 1 cụm
                    from scipy.spatial.distance import cdist
                    
                    # Tính trung bình của mỗi cụm
                    cluster_centers = {}
                    for label_id in set(labels):
                        if label_id != -1:
                            mask = labels == label_id
                            cluster_centers[label_id] = np.mean(embeddings_array[mask], axis=0)
                    
                    # Gán điểm nhiễu vào cụm gần nhất
                    noise_indices = np.where(labels == -1)[0]
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
        except ImportError:
            print("Thư viện HDBSCAN không có sẵn. Sử dụng KMeans thay thế.")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)
    
    # Thống kê kết quả phân cụm
    if labels is not None:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"\nKết quả phân cụm: {n_clusters} cụm")
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label == -1:
                print(f"Noise: {count} embeddings")
            else:
                print(f"Cluster {label}: {count} embeddings")
    
    return labels

def save_face_clusters(view, embedding_data, labels, output_dir="/fiftyone/data", prefix=""):
    """Lưu thông tin phân cụm khuôn mặt"""
    print("\n=== Lưu thông tin phân cụm khuôn mặt ===")
    
    if labels is None:
        print("Không có kết quả phân cụm để lưu")
        return
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo cấu trúc dữ liệu để lưu
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"{prefix}face_clusters_{timestamp}.json")
    
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[str(label)] = []
        
        sample_id = embedding_data["sample_ids"][idx]
        sample = view[sample_id]
        
        # Thêm thông tin khuôn mặt vào cụm
        clusters[str(label)].append({
            "sample_id": sample_id,
            "filepath": sample.filepath,
            "frame_number": embedding_data["frame_numbers"][idx],
            "bbox": embedding_data["bbox_info"][idx]
        })
    
    # Thêm trường cluster_id vào dataset
    print("Đang cập nhật trường cluster_id trong dataset...")
    label_field = f"{prefix}face_cluster_id"
    
    for idx, label in enumerate(labels):
        sample_id = embedding_data["sample_ids"][idx]
        sample = view[sample_id]
        
        # Tìm embedding tương ứng trong sample
        if sample.has_field("face_embeddings"):
            for i, fe in enumerate(sample.face_embeddings):
                # Kiểm tra nếu đây là embedding đang xét
                for face_idx in embedding_data["sample_to_face_map"].get(sample_id, []):
                    if face_idx == idx:
                        # Gán cluster_id vào fe
                        fe["cluster_id"] = int(label)
        
        # Lưu sample
        sample.save()
    
    # Lưu kết quả phân cụm
    print(f"Lưu thông tin phân cụm vào {result_file}...")
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_embeddings": len(embedding_data["embeddings"]),
            "num_clusters": len(clusters),
            "clusters": clusters
        }, f, indent=2)
    
    print(f"Đã lưu thông tin phân cụm thành công!")
    return result_file

def export_face_thumbnails(view, embedding_data, output_dir="/fiftyone/data/face_thumbnails"):
    """Xuất thumbnails của các khuôn mặt để kiểm tra trực quan"""
    print("\n=== Xuất thumbnails khuôn mặt ===")
    
    import cv2
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Kiểm tra số lượng khuôn mặt, giới hạn nếu quá nhiều
    num_faces = len(embedding_data["embeddings"])
    max_faces = 100
    if num_faces > max_faces:
        print(f"Có {num_faces} khuôn mặt, sẽ xuất tối đa {max_faces} thumbnails...")
        # Chọn ngẫu nhiên các chỉ số
        indices = np.random.choice(num_faces, max_faces, replace=False)
    else:
        print(f"Xuất {num_faces} thumbnails khuôn mặt...")
        indices = range(num_faces)
    
    exported_faces = 0
    for idx in tqdm(indices):
        sample_id = embedding_data["sample_ids"][idx]
        sample = view[sample_id]
        
        try:
            # Đọc ảnh gốc
            image = cv2.imread(sample.filepath)
            if image is None:
                continue
            
            # Lấy bounding box
            bbox = embedding_data["bbox_info"][idx]
            height, width = image.shape[:2]
            
            # Chuyển sang tọa độ pixel
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            w = int(bbox[2] * width)
            h = int(bbox[3] * height)
            
            # Cắt vùng khuôn mặt
            face_img = image[y1:y1+h, x1:x1+w]
            if face_img.size == 0:
                continue
            
            # Tạo tên file
            frame_num = embedding_data["frame_numbers"][idx]
            output_file = os.path.join(output_dir, f"face_{os.path.basename(sample.filepath)}_{frame_num}_{idx}.jpg")
            
            # Lưu ảnh
            cv2.imwrite(output_file, face_img)
            exported_faces += 1
            
        except Exception as e:
            print(f"Lỗi khi xuất face thumbnail cho {sample.filepath}: {e}")
    
    print(f"Đã xuất {exported_faces} thumbnails khuôn mặt vào {output_dir}")
    return output_dir

def create_character_field(view, embedding_data, labels):
    """Tạo trường character_detections dựa trên phân cụm"""
    print("\n=== Tạo trường character_detections dựa trên phân cụm ===")
    
    if labels is None:
        print("Không có kết quả phân cụm để tạo character_detections")
        return
    
    # Map từ cluster_id sang màu
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values())
    character_colors = {}
    for label in set(labels):
        if label == -1:  # Noise
            character_colors[label] = "#777777"  # Xám
        else:
            color_idx = label % len(colors)
            character_colors[label] = colors[color_idx]
    
    # Cập nhật character_detections trong dataset
    print("Đang tạo character_detections...")
    
    for idx, label in enumerate(labels):
        sample_id = embedding_data["sample_ids"][idx]
        sample = view[sample_id]
        
        # Bỏ qua nếu là noise
        if label == -1:
            continue
        
        # Lấy bounding box
        bbox = embedding_data["bbox_info"][idx]
        
        # Kiểm tra nếu đã có character_detections
        if not sample.has_field("character_detections"):
            sample["character_detections"] = fo.Detections()
        
        # Tạo detection cho character
        detection = fo.Detection(
            label=f"Character_{label}",
            bounding_box=bbox,
            confidence=1.0,
            index=idx,
            character_id=int(label),
            color=character_colors[label]
        )
        
        # Thêm vào detections
        sample["character_detections"].detections.append(detection)
        
        # Lưu sample
        sample.save()
    
    print("Đã tạo character_detections thành công!")

def launch_app_with_view(view, options=None):
    """Khởi chạy FiftyOne App với view cụ thể"""
    if options is None:
        options = {}
    
    # Tạo view dựa vào options
    if "sort_by" in options:
        view = view.sort_by(options["sort_by"], reverse=options.get("reverse", False))
    
    # Lưu view nếu cần
    if "view_name" in options:
        view_name = options["view_name"]
        view_collection = fo.load_dataset(view.dataset._doc.name).save_view(
            view_name, view, overwrite=True
        )
        print(f"Đã lưu view '{view_name}'")
    
    # Khởi chạy app
    session = fo.launch_app(view)
    print(f"\nĐã mở FiftyOne App với {len(view)} samples")
    
    # Thêm hướng dẫn tùy thuộc vào options
    if "brain_key" in options:
        print(f"Trong Embeddings Panel, chọn visualization key: {options['brain_key']}")
    
    if "label_field" in options:
        print(f"Chọn 'color by: {options['label_field']}' để xem phân cụm theo màu")
    
    return session

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description="Phân tích embeddings khuôn mặt với FiftyOne Brain")
    parser.add_argument("video_name", help="Tên video để phân tích (ví dụ: 'test', 'test2')")
    parser.add_argument("--dataset", default="video_dataset_faces_dlib", 
                        help="Tên dataset chứa embeddings khuôn mặt")
    parser.add_argument("--embedding-field", default="face_embeddings", 
                        choices=["face_embeddings", "dlib_face_embeddings"],
                        help="Trường chứa embeddings khuôn mặt")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Ngưỡng confidence tối thiểu cho embeddings")
    parser.add_argument("--method", choices=["all", "similarity", "uniqueness", "visualization", "clustering"],
                        default="all", help="Phương pháp phân tích (mặc định: all)")
    parser.add_argument("--viz-method", choices=["umap", "tsne", "pca"],
                        default="umap", help="Phương pháp trực quan hóa (mặc định: umap)")
    parser.add_argument("--cluster-method", choices=["kmeans", "agglomerative", "dbscan", "hdbscan"],
                        default="hdbscan", help="Phương pháp phân cụm (mặc định: hdbscan)")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Số cụm cho phân cụm (mặc định: tự động ước tính)")
    parser.add_argument("--export-thumbnails", action="store_true",
                        help="Xuất thumbnails khuôn mặt")
    parser.add_argument("--create-characters", action="store_true",
                        help="Tạo trường character_detections dựa trên phân cụm")
    
    args = parser.parse_args()
    
    # Thiết lập kết nối
    setup_connection()
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    if dataset is None:
        return
    
    # Lấy frames của video
    view = get_video_frames(dataset, args.video_name)
    if len(view) == 0:
        return
    
    # Thu thập embeddings
    embedding_data = collect_embeddings(view, args.embedding_field, args.min_confidence)
    if not embedding_data["embeddings"]:
        return
    
    # Brain key prefix
    brain_key_prefix = f"{args.video_name}_{args.embedding_field}"
    
    # Thực hiện các phương pháp phân tích
    similarity_results = None
    uniqueness_results = None
    visualization_results = None
    clustering_labels = None
    
    if args.method in ["all", "similarity"]:
        similarity_results = compute_similarity(
            view, embedding_data, 
            brain_key=f"{brain_key_prefix}_similarity"
        )
    
    if args.method in ["all", "uniqueness"]:
        uniqueness_results = compute_uniqueness(
            view, embedding_data,
            brain_key=f"{brain_key_prefix}_uniqueness"
        )
    
    if args.method in ["all", "visualization"]:
        visualization_results = compute_visualization(
            view, embedding_data, 
            method=args.viz_method,
            brain_key=f"{brain_key_prefix}_viz_{args.viz_method}",
            label_field=f"{args.video_name}_face_cluster"
        )
    
    if args.method in ["all", "clustering"]:
        clustering_labels = cluster_faces(
            embedding_data, 
            method=args.cluster_method,
            n_clusters=args.n_clusters
        )
        
        # Lưu kết quả phân cụm
        if clustering_labels is not None:
            save_face_clusters(
                view, embedding_data, clustering_labels,
                prefix=f"{args.video_name}_{args.cluster_method}_"
            )
    
    # Xuất thumbnails khuôn mặt nếu yêu cầu
    if args.export_thumbnails:
        export_face_thumbnails(view, embedding_data)
    
    # Tạo character_detections nếu yêu cầu
    if args.create_characters and clustering_labels is not None:
        create_character_field(view, embedding_data, clustering_labels)
    
    # Khởi chạy app với view phù hợp
    app_options = {}
    
    if args.method == "similarity" and similarity_results:
        app_options["sort_by"] = f"similarity_{similarity_results['target_id']}"
        app_options["reverse"] = True
        app_options["view_name"] = f"{args.video_name}_similar_faces"
    elif args.method == "uniqueness" and uniqueness_results:
        app_options["sort_by"] = "uniqueness"
        app_options["reverse"] = True
        app_options["view_name"] = f"{args.video_name}_unique_faces"
    elif args.method == "visualization" and visualization_results:
        app_options["brain_key"] = visualization_results["brain_key"]
        app_options["label_field"] = visualization_results["label_field"]
        app_options["view_name"] = f"{args.video_name}_visualization"
    
    # Khởi chạy app
    session = launch_app_with_view(view, app_options)
    
    print("\nTruy cập http://localhost:5151 để xem kết quả")

if __name__ == "__main__":
    main()