import fiftyone as fo
import numpy as np
from sklearn.cluster import HDBSCAN # Hoặc from sklearn.cluster import KMeans
import fiftyone.brain as fob # Để sử dụng các hàm tiện ích nếu cần

def cluster_characters_globally(dataset_name, detections_field, embedding_field_name, new_cluster_id_field):
    """
    Gom nhóm các nhân vật (detections) trên toàn bộ dataset dựa trên embeddings của chúng.

    Args:
        dataset_name (str): Tên của dataset FiftyOne.
        detections_field (str): Tên trường chứa danh sách các detections (ví dụ: "dlib_face_detections").
        embedding_field_name (str): Tên trường bên trong mỗi detection chứa embedding (ví dụ: "custom_embedding").
        new_cluster_id_field (str): Tên trường mới để lưu ID cluster toàn cục cho mỗi detection.
    """
    print(f"--- Bắt đầu gom nhóm nhân vật toàn cục cho dataset: {dataset_name} ---")

    # 1. Kết nối và Tải Dataset
    if not fo.dataset_exists(dataset_name):
        print(f"Lỗi: Dataset '{dataset_name}' không tồn tại.")
        return
    dataset = fo.load_dataset(dataset_name)
    print(f"Đã tải dataset '{dataset_name}' với {len(dataset)} samples.")

    # 2. Thu thập Embeddings và IDs
    all_embeddings = []
    detection_object_ids = [] # Lưu ObjectId của từng detection
    sample_ids_for_detections = [] # Lưu sample_id chứa detection đó

    print(f"Đang thu thập embeddings từ trường '{detections_field}.detections.{embedding_field_name}'...")
    # Sử dụng .select_fields() để chỉ lấy các trường cần thiết, giúp tăng tốc độ
    view = dataset.select_fields([detections_field])

    for sample in view.iter_samples(autosave=False, progress=True):
        # Trường detections_field có thể không tồn tại trên mọi sample
        detections_obj = sample.get_field(detections_field)
        if detections_obj and detections_obj.detections:
            for detection in detections_obj.detections:
                # Đảm bảo detection có embedding
                if hasattr(detection, embedding_field_name) and getattr(detection, embedding_field_name) is not None:
                    all_embeddings.append(getattr(detection, embedding_field_name))
                    detection_object_ids.append(detection._id) # Lưu ObjectId của detection
                    sample_ids_for_detections.append(sample.id) # Lưu sample_id để truy cập lại sample
                else:
                    print(f"Cảnh báo: Detection ID {detection._id} trong sample {sample.id} không có trường '{embedding_field_name}'.")

    if not all_embeddings:
        print("Không tìm thấy embeddings nào để gom nhóm. Kết thúc.")
        return

    print(f"Đã thu thập được {len(all_embeddings)} embeddings từ {len(set(sample_ids_for_detections))} samples.")
    embeddings_matrix = np.array(all_embeddings)
    print(f"Shape của embeddings_matrix: {embeddings_matrix.shape}")

    # 3. Gom Nhóm (Clustering)
    # Sử dụng HDBSCAN
    # Bạn có thể điều chỉnh các tham số này:
    # min_cluster_size: Số lượng điểm tối thiểu để tạo thành một cluster.
    # min_samples: Số lượng láng giềng tối thiểu của một core point.
    # allow_single_cluster: Cho phép chỉ có một cluster duy nhất nếu dữ liệu là như vậy.
    print("Đang thực hiện gom nhóm với HDBSCAN...")
    clusterer = HDBSCAN(min_cluster_size=5, min_samples=None, metric='euclidean', allow_single_cluster=True)
    # clusterer = KMeans(n_clusters=10, random_state=42) # Ví dụ với KMeans

    cluster_labels = clusterer.fit_predict(embeddings_matrix)
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise_points = np.sum(cluster_labels == -1)
    print(f"HDBSCAN hoàn tất. Tìm thấy {num_clusters} clusters và {num_noise_points} điểm nhiễu.")
    print(f"Phân phối các nhãn cluster (bao gồm nhiễu -1): {np.unique(cluster_labels, return_counts=True)}")


    # 4. Lưu trữ global_character_id
    print(f"Đang lưu trữ ID cluster vào trường '{new_cluster_id_field}' cho mỗi detection...")

    # Cần phải lặp lại qua dataset để cập nhật các detections
    # Cách hiệu quả hơn là lưu trữ một map từ detection._id -> cluster_label
    # sau đó dùng bulk_write nếu có nhiều. Hoặc lặp qua các sample đã có detection.

    # Tạo một dictionary để map ObjectId của detection với sample_id và index của nó trong sample đó
    # Điều này không thực sự cần thiết nếu chúng ta cập nhật trực tiếp qua sample_ids_for_detections và detection_object_ids
    
    # Lặp lại qua dataset gốc để cập nhật
    # Điều này có thể chậm nếu dataset lớn. Xem xét các cách tối ưu hơn nếu cần.
    # Cách tối ưu hơn: sử dụng các ID đã lưu để chỉ tải các sample cần thiết.
    
    # Lấy các sample chứa các detection cần cập nhật
    samples_to_update_view = dataset.select(list(set(sample_ids_for_detections)))

    # Tạo một map từ detection_id sang cluster_label
    detection_id_to_cluster_label = {det_id: label for det_id, label in zip(detection_object_ids, cluster_labels)}

    with fo.ProgressBar(total=len(samples_to_update_view)) as pb:
        for sample in samples_to_update_view: # Không dùng autosave ở đây, lưu sau khi cập nhật hết 1 sample
            modified_in_sample = False
            detections_obj = sample.get_field(detections_field) # Lấy lại object Detections
            if detections_obj and detections_obj.detections:
                for detection in detections_obj.detections:
                    if detection._id in detection_id_to_cluster_label:
                        cluster_id = int(detection_id_to_cluster_label[detection._id]) # Đảm bảo là int
                        setattr(detection, new_cluster_id_field, cluster_id)
                        modified_in_sample = True
            if modified_in_sample:
                sample.save() # Lưu sample nếu có thay đổi
            pb.update()

    print(f"Đã cập nhật trường '{new_cluster_id_field}' cho các detections.")
    print("--- Gom nhóm nhân vật toàn cục hoàn tất ---")

    # (Tùy chọn) Thêm tag cho các detections nhiễu
    # view_noise = dataset.filter_labels(f"{detections_field}.detections", F(new_cluster_id_field) == -1)
    # view_noise.tag_labels("noise_character")
    # print(f"Đã tag {len(view_noise)} detections nhiễu với tag 'noise_character'")

    # (Tùy chọn) In ra một số thông tin về các cluster
    # for cluster_id_val in set(cluster_labels):
    #     if cluster_id_val != -1:
    #         count = np.sum(cluster_labels == cluster_id_val)
    #         print(f"Cluster {cluster_id_val}: {count} detections")


if __name__ == '__main__':
    # Cấu hình MongoDB URI nếu cần
    # fo.config.database_uri = "mongodb://localhost:27017"

    dataset_to_process = "video_dataset_faces_dlib"
    # Dựa trên ảnh bạn cung cấp:
    # - Trường chứa object Detections (cha của list 'detections') là 'dlib_face_detections'
    # - Bên trong mỗi object Detection, trường embedding là 'custom_embedding'
    detections_field_path = "dlib_face_detections"
    embedding_field_in_detection = "custom_embedding"
    new_cluster_field = "global_character_id" # Tên trường mới bạn muốn tạo

    cluster_characters_globally(dataset_to_process, detections_field_path, embedding_field_in_detection, new_cluster_field)

    # Sau khi chạy, bạn có thể mở FiftyOne App để xem kết quả:
    # session = fo.launch_app(fo.load_dataset(dataset_to_process))
    # Bạn có thể tô màu các bounding box dựa trên trường 'global_character_id' mới này.