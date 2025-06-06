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
import gc

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

def cluster_faces_dbscan(face_embeddings, eps=None, min_samples=5):
    """Nhóm khuôn mặt bằng DBSCAN - Phiên bản cải tiến"""
    from sklearn.cluster import DBSCAN
    
    if not face_embeddings:
        print("Không có embedding nào để nhóm.")
        return []

    embeddings_array = np.array(face_embeddings)
    print(f"Shape của embedding array: {embeddings_array.shape}")
    
    # Chuẩn hóa embeddings
    embeddings_array = normalize(embeddings_array)
    
    # Tìm giá trị epsilon tối ưu
    if eps is None:
        from sklearn.neighbors import NearestNeighbors
        # Sử dụng khoảng cách cosine thay vì Euclidean
        nbrs = NearestNeighbors(n_neighbors=min(20, len(embeddings_array)-1), metric='cosine')
        nbrs.fit(embeddings_array)
        distances, _ = nbrs.kneighbors(embeddings_array)
        
        # Sắp xếp khoảng cách đến láng giềng thứ k
        k_distances = np.sort(distances[:, -1])
        
        # Thử nhiều giá trị epsilon khác nhau
        eps_candidates = [
            np.percentile(k_distances, 10),  # Rất chặt - nhiều cụm nhỏ
            np.percentile(k_distances, 20),  # Chặt
            np.percentile(k_distances, 30),  # Trung bình thấp
            np.percentile(k_distances, 50),  # Trung bình
        ]
        
        best_eps = None
        best_n_clusters = 0
        best_labels = None
        best_score = -1
        
        print("Thử nghiệm các giá trị epsilon khác nhau:")
        for candidate_eps in eps_candidates:
            # Chạy DBSCAN với epsilon ứng viên
            temp_dbscan = DBSCAN(eps=candidate_eps, min_samples=min_samples, metric='cosine')
            temp_labels = temp_dbscan.fit_predict(embeddings_array)
            
            # Đếm số lượng cụm (không tính nhiễu)
            temp_n_clusters = len(set(temp_labels)) - (1 if -1 in temp_labels else 0)
            temp_n_noise = list(temp_labels).count(-1)
            temp_noise_ratio = temp_n_noise / len(temp_labels) * 100
            
            print(f"  Với epsilon={candidate_eps:.4f}: {temp_n_clusters} cụm, {temp_n_noise} điểm nhiễu ({temp_noise_ratio:.1f}%)")
            
            # Tính silhouette score nếu có nhiều hơn 1 cụm và không quá nhiều nhiễu
            if temp_n_clusters > 1 and temp_noise_ratio < 50:
                # Loại bỏ điểm nhiễu khi tính silhouette
                mask = temp_labels != -1
                if np.sum(mask) > temp_n_clusters:
                    from sklearn.metrics import silhouette_score
                    try:
                        score = silhouette_score(embeddings_array[mask], temp_labels[mask], metric='cosine')
                        print(f"    Silhouette score: {score:.4f}")
                        
                        # Cập nhật nếu đây là kết quả tốt nhất
                        if score > best_score:
                            best_score = score
                            best_eps = candidate_eps
                            best_n_clusters = temp_n_clusters
                            best_labels = temp_labels
                    except:
                        pass
            
            # Nếu chưa có kết quả tốt và đây là kết quả đầu tiên với nhiều hơn 1 cụm
            if best_eps is None and temp_n_clusters > 1 and temp_noise_ratio < 70:
                best_eps = candidate_eps
                best_n_clusters = temp_n_clusters
                best_labels = temp_labels
        
        # Nếu tìm được epsilon phù hợp
        if best_eps is not None:
            eps = best_eps
            print(f"Chọn epsilon = {eps:.4f} với {best_n_clusters} cụm")
            # Sử dụng kết quả đã tính toán
            labels = best_labels
            
            # Kiểm tra số lượng cụm và điểm nhiễu
            n_clusters = best_n_clusters
            n_noise = list(labels).count(-1)
            print(f"DBSCAN tìm thấy {n_clusters} cụm và {n_noise} điểm nhiễu ({n_noise/len(labels)*100:.1f}%)")
            
            # In thông tin phân bố cụm
            unique_labels, counts = np.unique(labels, return_counts=True)
            print("Phân bố kích thước clusters:")
            for label, count in zip(unique_labels, counts):
                if label == -1:
                    print(f"Noise: {count} samples")
                else:
                    print(f"Cluster {label}: {count} samples")
        else:
            # Nếu không tìm được epsilon tốt, thử giá trị nhỏ hơn
            eps = np.percentile(k_distances, 5)
            print(f"Không tìm thấy epsilon tốt, thử nghiệm với giá trị nhỏ: {eps:.4f}")
            
            # Chạy DBSCAN với epsilon nhỏ
            dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine')
            labels = dbscan.fit_predict(embeddings_array)
    else:
        # Nếu eps được cung cấp, sử dụng nó
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(embeddings_array)
        
        # Kiểm tra số lượng cụm và điểm nhiễu
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"DBSCAN tìm thấy {n_clusters} cụm và {n_noise} điểm nhiễu ({n_noise/len(labels)*100:.1f}%)")
        
        # In thông tin phân bố cụm
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Phân bố kích thước clusters:")
        for label, count in zip(unique_labels, counts):
            if label == -1:
                print(f"Noise: {count} samples")
            else:
                print(f"Cluster {label}: {count} samples")
    
    # Xử lý điểm nhiễu (gán vào cụm gần nhất)
    if -1 in labels:
        print("\nĐang gán lại các điểm nhiễu vào cụm gần nhất...")
        noise_indices = np.where(labels == -1)[0]
        valid_indices = np.where(labels != -1)[0]
        
        # Nếu tất cả đều là nhiễu, tạo một cụm mới
        if len(valid_indices) == 0:
            print("Tất cả các điểm đều là nhiễu, tạo một cụm duy nhất")
            labels = np.zeros(len(labels), dtype=int)
        else:
            # Tính khoảng cách từ mỗi điểm nhiễu đến các tâm cụm
            from scipy.spatial.distance import cdist
            
            # Tính tọa độ tâm của mỗi cụm
            cluster_centers = {}
            for label_id in set(labels):
                if label_id != -1:
                    mask = labels == label_id
                    cluster_centers[label_id] = np.mean(embeddings_array[mask], axis=0)
            
            # Gán mỗi điểm nhiễu vào cụm gần nhất
            for idx in noise_indices:
                noise_point = embeddings_array[idx].reshape(1, -1)
                min_dist = float('inf')
                closest_cluster = 0
                
                for label_id, center in cluster_centers.items():
                    dist = cdist(noise_point, center.reshape(1, -1), metric='cosine')
                    if dist < min_dist:
                        min_dist = dist
                        closest_cluster = label_id
                
                labels[idx] = closest_cluster
        
        # In thông tin sau khi gán lại
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Phân bố sau khi gán điểm nhiễu:")
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples")
    
    return labels

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
    
    # Thiết lập min_samples nếu không được cung cấp, make it more robust for small min_cluster_size
    if min_samples is None:
        if min_cluster_size <= 1: # Allow min_samples to be 1 if min_cluster_size is 1
            min_samples = 1
        elif min_cluster_size <= 3: # For small min_cluster_size (e.g. 2 or 3)
            min_samples = max(1, min_cluster_size -1) # Try min_samples = 1 or 2
        else: # Default logic for larger min_cluster_size
            min_samples = max(2, min_cluster_size // 2) 

    print(f"HDBSCAN (Pass 1) với tham số: min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric='euclidean', allow_single_cluster=True (EOM selection)")
    
    # Cluster sử dụng HDBSCAN với tham số chính xác - PASS 1
    clusterer_pass1 = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        allow_single_cluster=True, 
        prediction_data=True # Keep for potential later use, though primary use is for fit_predict here
    )
    
    labels = clusterer_pass1.fit_predict(embeddings_array)
    
    n_clusters_pass1 = len(set(l for l in labels if l != -1))
    n_noise_pass1 = list(labels).count(-1)
    print(f"HDBSCAN (Pass 1) tìm thấy {n_clusters_pass1} cụm và {n_noise_pass1} điểm nhiễu ({n_noise_pass1/len(labels)*100:.1f}% nếu len > 0 else 0.0%)")

    unique_labels_pass1, counts_pass1 = np.unique(labels, return_counts=True)
    print("Phân bố kích thước clusters (Pass 1):")
    for label, count in zip(unique_labels_pass1, counts_pass1):
        if label == -1:
            print(f"  Noise: {count} samples")
        else:
            print(f"  Cluster {label}: {count} samples")

    # --- Secondary Clustering on Noise from Pass 1 ---
    MIN_POINTS_FOR_SECONDARY_CLUSTERING = max(5, min_cluster_size * 2) # Heuristic: at least e.g. 5-10 points
    # Parameters for secondary clustering - should be more sensitive
    min_cluster_size_secondary = max(2, min_cluster_size // 2 if min_cluster_size > 2 else 2) 
    min_samples_secondary = max(1, min_cluster_size_secondary // 2 if min_cluster_size_secondary > 1 else 1)

    if n_noise_pass1 >= MIN_POINTS_FOR_SECONDARY_CLUSTERING and n_clusters_pass1 > 0 : # Only run if there's substantial noise AND some initial clusters were found
        print(f"\nHDBSCAN (Pass 2) - Chạy lại trên {n_noise_pass1} điểm nhiễu từ Pass 1.")
        print(f"  Tham số Pass 2: min_cluster_size={min_cluster_size_secondary}, min_samples={min_samples_secondary}")
        
        noise_indices_pass1 = np.where(labels == -1)[0]
        noise_embeddings_pass1 = embeddings_array[noise_indices_pass1]

        clusterer_pass2 = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size_secondary,
            min_samples=min_samples_secondary,
            metric='euclidean',
            allow_single_cluster=True 
            # prediction_data for pass 2 might not be needed if we don't recurse noise further
        )
        labels_pass2_raw = clusterer_pass2.fit_predict(noise_embeddings_pass1)
        
        n_clusters_pass2 = len(set(l for l in labels_pass2_raw if l != -1))
        n_noise_pass2 = list(labels_pass2_raw).count(-1)
        print(f"HDBSCAN (Pass 2) tìm thấy {n_clusters_pass2} cụm mới và {n_noise_pass2} điểm nhiễu còn lại.")

        if n_clusters_pass2 > 0:
            # Re-index labels from pass 2 so they don't clash with pass 1 labels
            max_label_pass1 = -1
            if n_clusters_pass1 > 0: # Should be true due to condition above
                 max_label_pass1 = np.max(labels[labels != -1])
            
            current_new_label_start = max_label_pass1 + 1
            
            # Create a mapping for pass 2 labels
            labels_pass2_remapped = np.copy(labels_pass2_raw)
            unique_raw_pass2_labels = sorted(list(set(l for l in labels_pass2_raw if l != -1))) # Get unique cluster IDs from pass2
            
            map_pass2_to_final = {}
            for raw_p2_label in unique_raw_pass2_labels:
                map_pass2_to_final[raw_p2_label] = current_new_label_start
                current_new_label_start += 1
            
            # Apply the mapping
            for i in range(len(labels_pass2_raw)):
                if labels_pass2_raw[i] != -1:
                    labels_pass2_remapped[i] = map_pass2_to_final[labels_pass2_raw[i]]
                else:
                    labels_pass2_remapped[i] = -1 # Noise from pass 2 remains -1

            # Update the main 'labels' array with results from pass 2
            for i, original_noise_idx in enumerate(noise_indices_pass1):
                labels[original_noise_idx] = labels_pass2_remapped[i]
            
            print("Phân bố kích thước clusters (Sau Pass 2 gộp vào):")
            unique_labels_combined, counts_combined = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels_combined, counts_combined):
                if label == -1:
                    print(f"  Noise (còn lại): {count} samples")
                else:
                    print(f"  Cluster {label}: {count} samples")
        else:
            print("Pass 2 không tìm thấy cụm mới nào trong nhiễu của Pass 1.")
    elif n_clusters_pass1 == 0 and n_noise_pass1 > 0:
        print("\nHDBSCAN (Pass 1) không tìm thấy cụm nào, tất cả là nhiễu. Sẽ được xử lý sau.")
    
    # At this point, 'labels' contains results from Pass 1 and potentially refined by Pass 2.
    # 'n_clusters' should reflect the total number of unique nonnoise clusters found.
    n_clusters = len(set(l for l in labels if l != -1))
    
    # Xử lý điểm nhiễu còn lại (sau Pass 1 và Pass 2) sử dụng approximate_predict hoặc cdist
    # For simplicity now, let's skip approximate_predict as it was causing warnings and might not be ideal for multiple passes.
    # We will directly go to cdist based reassignment for any remaining -1.
    
    if -1 in labels:
        print("\nĐang gán lại các điểm nhiễu còn lại vào cụm gần nhất (dựa trên tâm cụm)...")
        noise_indices_remaining = np.where(labels == -1)[0]
        valid_indices = np.where(labels != -1)[0]
        
        if len(valid_indices) > 0 and len(noise_indices_remaining) > 0:
            cluster_centers = {}
            current_valid_cluster_ids = sorted(list(set(l for l in labels if l != -1))) # Get current valid cluster IDs
            
            for label_id in current_valid_cluster_ids:
                mask = labels == label_id
                if np.sum(mask) > 0: 
                    cluster_centers[label_id] = np.mean(embeddings_array[mask], axis=0)
            
            if cluster_centers: 
                from scipy.spatial.distance import cdist
                for idx in noise_indices_remaining:
                    noise_point = embeddings_array[idx].reshape(1, -1)
                    distances = {
                        cluster_id: cdist(noise_point, center.reshape(1, -1), metric='euclidean')[0][0]
                        for cluster_id, center in cluster_centers.items()
                    }
                    if distances: 
                        closest_cluster = min(distances, key=distances.get)
                        labels[idx] = closest_cluster
                    else: 
                        # This case should ideally not be reached if cluster_centers is populated.
                        # If for some reason it is, assign to the first valid cluster or 0.
                        labels[idx] = current_valid_cluster_ids[0] if current_valid_cluster_ids else 0
                
                print("Phân bố sau khi gán nốt điểm nhiễu:")
                unique_labels_final, counts_final = np.unique(labels, return_counts=True)
                for label, count in zip(unique_labels_final, counts_final):
                    print(f"  Cluster {label}: {count} samples")

        elif len(valid_indices) == 0 and len(noise_indices_remaining) > 0 : # All points are still noise
            print("Tất cả các điểm vẫn là nhiễu sau các bước, tạo một cụm duy nhất.")
            labels = np.zeros(len(labels), dtype=int)
            # n_clusters is updated below
        
    # Đảm bảo nhãn là các số nguyên liên tiếp từ 0
    # Recalculate n_clusters based on the final state of labels before re-mapping
    final_unique_non_noise_labels = sorted(list(set(l for l in labels if l != -1)))
    
    if not final_unique_non_noise_labels and np.all(labels == -1): # Handles case where all points ended up as noise
        labels = np.zeros(len(labels), dtype=int) # Assign all to cluster 0
        final_unique_non_noise_labels = [0]

    elif not final_unique_non_noise_labels and np.all(labels != -1): # All points in some cluster(s), but list is empty (e.g. single cluster 0)
         final_unique_non_noise_labels = sorted(list(set(labels)))


    mapping = {old_label: new_label for new_label, old_label in enumerate(final_unique_non_noise_labels)}
    
    final_labels_mapped = np.array([mapping.get(label, -1) for label in labels]) # Map to new, noise becomes -1 temporarily if not in mapping
                                                                               # then handle these -1s if any.
    
    # If any points were noise and not in final_unique_non_noise_labels, they'd be -1 from .get()
    # This shouldn't happen if the above logic (assigning all remaining noise to a cluster or to 0) worked.
    # But as a safeguard:
    if -1 in final_labels_mapped:
        # This implies some original -1 didn't get caught by reassignments and weren't in final_unique_non_noise_labels
        # Assign them to the largest cluster or cluster 0
        if final_unique_non_noise_labels: # If there were some valid clusters mapped
            # Find largest cluster among the mapped ones
            unique_mapped, counts_mapped = np.unique(final_labels_mapped[final_labels_mapped != -1], return_counts=True)
            if len(unique_mapped) > 0:
                largest_cluster_label = unique_mapped[np.argmax(counts_mapped)]
                final_labels_mapped[final_labels_mapped == -1] = largest_cluster_label
            else: # No valid clusters formed, everything was noise, map to 0
                 final_labels_mapped[final_labels_mapped == -1] = 0
        else: # No clusters formed at all, everything was noise, map all to 0
            final_labels_mapped = np.zeros(len(labels), dtype=int)


    labels = final_labels_mapped
    
    return labels

def group_faces_into_characters():
    """Nhóm các khuôn mặt trong dataset thành các nhân vật theo từng video riêng biệt"""
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Load dataset khuôn mặt
    face_dataset_name = "video_dataset_faces_dlib_test"
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
                print(f"Không xác định được video nguồn cho {sample.id} (filepath: {sample.filepath})")
                # Nếu không xác định được video_id chính xác, tạo một ID dựa trên filepath
                if hasattr(sample, "filepath"):
                    parts = sample.filepath.split('/')
                    potential_video_id = None
                    for j, part in enumerate(parts):
                        if part == "frames" and j < len(parts) - 1:
                            potential_video_id = parts[j+1]
                            break
                    if potential_video_id:
                        video_id = potential_video_id
                        # Thêm video vào videos_map nếu cần
                        if video_id not in videos_map:
                            videos_map[video_id] = {
                                "name": f"{video_id}.mp4",
                                "id": video_id,
                                "fps": 24  # Giá trị mặc định
                            }
                
                if video_id is None:
                    continue
                
            # Thêm frame vào danh sách của video tương ứng
            if video_id not in frames_by_video:
                frames_by_video[video_id] = []
                
            frames_by_video[video_id].append(sample.id)
    
    print(f"Tìm thấy {len(frames_by_video)} videos có frames với face embeddings")
    
    # Lưu trữ kết quả phân nhóm nhân vật cho tất cả video
    all_character_segments = {}  # video_id -> character_id -> segments
    
    # Xử lý từng video riêng biệt - đảm bảo thuật toán HDBSCAN áp dụng riêng cho từng video
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
        
        # Tự động ước tính số nhân vật tối ưu cho video này
        print(f"\nBước 1: Ước tính số nhân vật trong video {video_id}")
        print("-" * 50)
        n_clusters = estimate_optimal_clusters(all_embeddings)
        
        # Nhóm các khuôn mặt bằng HDBSCAN cho video này
        print(f"\nBước 2: Nhóm các khuôn mặt trong video {video_id} bằng HDBSCAN")
        print("-" * 50)
        
        # Dynamic min_cluster_size adjustment - Further refinement
        num_video_embeddings = len(all_embeddings)
        if num_video_embeddings == 0: 
            min_cluster_size_video = 1
        elif num_video_embeddings < 10: 
            min_cluster_size_video = max(1, num_video_embeddings // 3 if num_video_embeddings > 2 else 1) 
        elif num_video_embeddings < 30: 
            min_cluster_size_video = max(2, num_video_embeddings // 5)
        elif num_video_embeddings < 80: # e.g., 30-79 embeddings (covers 75 from log)
            # Let's be more aggressive here for smaller sets like 75.
            # Old: max(2, num_video_embeddings // 10) -> for 75 was 7
            min_cluster_size_video = max(2, num_video_embeddings // 20) # For 75 -> max(2,3) = 3. For 30 -> max(2,1) = 2.
        elif num_video_embeddings < 200:
            # Old: max(3, num_video_embeddings // 20)
            min_cluster_size_video = max(3, num_video_embeddings // 25) # For 100 -> 4. For 199 -> 7
        else: # For larger datasets
            # Old: max(5, num_video_embeddings // 30)
            min_cluster_size_video = max(5, num_video_embeddings // 40) # For 200 -> 5. For 400 -> 10
        
        if num_video_embeddings > 0 :
            min_cluster_size_video = max(1, min_cluster_size_video)
        else: 
             min_cluster_size_video = 1

        print(f"Sử dụng min_cluster_size động = {min_cluster_size_video} cho {num_video_embeddings} embeddings")
        
        try:
            # Thử với tham số giống code cũ
            labels = cluster_faces_hdbscan(
                all_embeddings, 
                min_cluster_size=min_cluster_size_video
                # min_samples will be determined inside cluster_faces_hdbscan based on min_cluster_size_video
            )
            
            # Nếu không tìm thấy cụm nào (hoặc chỉ 1 cụm nếu allow_single_cluster=False and it's all noise),
            # và nhiều hơn 1 embedding (để tránh lỗi với KMeans)
            # N_clusters from hdbscan includes noise as -1. So set(labels) might be {-1, 0, 1} for 2 clusters + noise.
            # We want to check if hdbscan found *meaningful* clusters.
            # len(set(l for l in labels if l != -1)) gives number of non-noise clusters.
            num_meaningful_clusters_hdbscan = len(set(l for l in labels if l != -1))

            if num_meaningful_clusters_hdbscan == 0 and len(all_embeddings) > n_clusters : # n_clusters is from estimate_optimal_clusters
                print(f"HDBSCAN không tìm thấy cụm ý nghĩa (found {num_meaningful_clusters_hdbscan}). Thử với phương pháp dự phòng KMeans với {n_clusters} cụm...")
                labels = cluster_faces_kmeans(all_embeddings, n_clusters)
            elif num_meaningful_clusters_hdbscan == 1 and len(set(labels)) > 1 and len(all_embeddings) > n_clusters:
                # This case means HDBSCAN might have found one cluster and some noise.
                # If KMeans suggests more clusters, it might be better.
                # Only switch if KMeans suggests more than 1 cluster.
                if n_clusters > 1:
                    print(f"HDBSCAN tìm thấy 1 cụm ý nghĩa nhưng KMeans đề xuất {n_clusters}. Thử KMeans.")
                    labels = cluster_faces_kmeans(all_embeddings, n_clusters)
                else:
                    print(f"HDBSCAN tìm thấy 1 cụm ý nghĩa. Giữ kết quả từ HDBSCAN.")

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

if __name__ == "__main__":
    print("Phân tích tự động số lượng nhân vật...")
    group_faces_into_characters()
