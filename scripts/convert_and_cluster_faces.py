import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.brain as fob
import numpy as np
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
EXPECTED_EMBEDDING_DIM = 128 # For Dlib face recognition model
# --- End Configuration ---

def convert_dlib_embeddings_to_detections(dataset, source_field="dlib_face_embeddings", target_field="dlib_face_detections"):
    """
    Chuyển đổi dlib_face_embeddings (list các dicts trên mỗi sample)
    thành một trường Detections trên mỗi sample.
    Mỗi Detection sẽ chứa bbox và embedding trong một trường tùy chỉnh.
    Cải thiện: Thêm kiểm tra và logging chi tiết hơn cho embeddings.
    """
    detections_created_count = 0
    problematic_samples_count = 0
    
    with fo.ProgressBar(total=len(dataset)) as pb:
        for sample_idx, sample in enumerate(pb(dataset)):
            sample_had_issues = False
            if not sample.has_field(source_field):
                # logger.debug(f"Sample {sample.id} không có trường nguồn '{source_field}'. Bỏ qua.")
                continue

            source_data = sample[source_field]
            if not source_data or not isinstance(source_data, list):
                # logger.warning(f"Sample {sample.id} có dữ liệu nguồn '{source_field}' không hợp lệ hoặc rỗng. Bỏ qua. Data: {source_data}")
                continue

            detections_list = []
            for face_idx, face_obj in enumerate(source_data):
                if not isinstance(face_obj, dict):
                    logger.warning(f"Sample {sample.id}, face object #{face_idx} trong '{source_field}' không phải là dict. Bỏ qua. Data: {face_obj}")
                    sample_had_issues = True
                    continue

                if "embedding" not in face_obj or "bbox" not in face_obj:
                    logger.warning(f"Sample {sample.id}, face object #{face_idx}: Thiếu 'embedding' hoặc 'bbox'. Bỏ qua. Keys: {face_obj.keys()}")
                    sample_had_issues = True
                    continue
                
                embedding_value = face_obj["embedding"]

                if embedding_value is None:
                    logger.warning(f"Sample {sample.id}, face object #{face_idx}: Giá trị 'embedding' là None. Bỏ qua embedding này.")
                    sample_had_issues = True
                    continue

                # Đảm bảo embedding_value là một list hoặc ndarray có thể lặp được
                if not isinstance(embedding_value, (list, np.ndarray)):
                    logger.warning(f"Sample {sample.id}, face object #{face_idx}: Kiểu dữ liệu của embedding không phải list/ndarray ({type(embedding_value)}). Bỏ qua embedding này.")
                    sample_had_issues = True
                    continue
                
                try:
                    # Chuyển đổi rõ ràng sang list các float
                    # Đảm bảo mỗi phần tử là float và là một list Python thuần túy
                    processed_embedding = [float(x) for x in embedding_value]
                except (TypeError, ValueError) as e:
                    logger.error(f"Sample {sample.id}, face object #{face_idx}: Lỗi khi chuyển đổi embedding sang list float: {e}. Embedding gốc: {embedding_value}. Bỏ qua embedding này.")
                    sample_had_issues = True
                    continue

                # Kiểm tra chiều dài của embedding
                if len(processed_embedding) != EXPECTED_EMBEDDING_DIM:
                    logger.warning(f"Sample {sample.id}, face object #{face_idx}: Chiều dài embedding ({len(processed_embedding)}) không khớp với mong đợi ({EXPECTED_EMBEDDING_DIM}). Bỏ qua embedding này.")
                    sample_had_issues = True
                    continue
                
                # Tạo detection
                try:
                    detection_to_add = fol.Detection(
                        label="dlib_face", # Hoặc lấy từ face_obj nếu có
                        bounding_box=face_obj["bbox"],
                        confidence=face_obj.get("confidence", 1.0) # dlib thường không có confidence
                    )
                    detection_to_add.custom_embedding = processed_embedding 
                    detections_list.append(detection_to_add)
                    detections_created_count +=1
                except Exception as e:
                    logger.error(f"Sample {sample.id}, face object #{face_idx}: Lỗi khi tạo Detection object: {e}. Bỏ qua detection này.")
                    sample_had_issues = True
            
            if detections_list:
                sample[target_field] = fol.Detections(detections=detections_list)
                sample.save()
            elif source_data: # có source_data nhưng không tạo được detection nào hợp lệ
                logger.warning(f"Sample {sample.id} có dữ liệu nguồn nhưng không tạo được detection hợp lệ nào từ đó.")
                sample_had_issues = True


            if sample_had_issues:
                problematic_samples_count +=1
    
    logger.info(f"Hoàn tất chuyển đổi. Tổng số detections được tạo với custom_embedding hợp lệ: {detections_created_count}")
    if problematic_samples_count > 0:
        logger.warning(f"Tìm thấy {problematic_samples_count} sample(s) có vấn đề trong quá trình chuyển đổi (chi tiết xem log ở trên).")


def verify_embeddings_consistency(dataset, detections_field="dlib_face_detections", embedding_attr="custom_embedding"):
    """
    Kiểm tra tính nhất quán của các embeddings đã được tạo trong trường Detections.
    """
    logger.info(f"\n--- Bắt đầu kiểm tra tính nhất quán của embeddings ---")
    logger.info(f"Dataset: {dataset.name}")
    logger.info(f"Kiểm tra trường: {detections_field}.detections.{embedding_attr}")

    total_detections_checked = 0
    detections_missing_embedding_attr = 0
    detections_with_none_embedding = 0
    embedding_lengths = {} # Lưu trữ tần suất của các độ dài embedding

    if not dataset.has_sample_field(detections_field):
        logger.error(f"Trường detections '{detections_field}' không tồn tại trong dataset. Không thể xác minh.")
        return

    with fo.ProgressBar(total=len(dataset)) as pb:
        for sample in pb(dataset.select_fields(detections_field)): # Chỉ load trường cần thiết
            if not sample[detections_field] or not sample[detections_field].detections:
                continue

            for detection in sample[detections_field].detections:
                total_detections_checked += 1
                if not hasattr(detection, embedding_attr):
                    detections_missing_embedding_attr += 1
                    continue
                
                emb = getattr(detection, embedding_attr)

                if emb is None:
                    detections_with_none_embedding += 1
                    continue
                
                if not isinstance(emb, (list, np.ndarray)):
                    logger.warning(f"Sample {sample.id}, Detection {detection.id}: embedding không phải list/ndarray ({type(emb)}).")
                    # Coi như đây là một dạng 'None' hoặc lỗi
                    detections_with_none_embedding +=1 
                    continue

                length = len(emb)
                embedding_lengths[length] = embedding_lengths.get(length, 0) + 1
    
    logger.info(f"Tổng số detections đã kiểm tra: {total_detections_checked}")
    if detections_missing_embedding_attr > 0:
        logger.warning(f"Số detections thiếu thuộc tính '{embedding_attr}': {detections_missing_embedding_attr}")
    if detections_with_none_embedding > 0:
        logger.warning(f"Số detections có '{embedding_attr}' là None hoặc kiểu không hợp lệ: {detections_with_none_embedding}")
    
    logger.info("Phân phối độ dài embedding tìm thấy:")
    if embedding_lengths:
        for length, count in embedding_lengths.items():
            logger.info(f"  Độ dài {length}: {count} lần")
        
        # Kiểm tra nếu có nhiều hơn 1 độ dài hoặc độ dài không như mong đợi
        if len(embedding_lengths) > 1:
            logger.error("LỖI TIỀM ẨN: Tìm thấy NHIỀU HƠN MỘT độ dài embedding. Điều này sẽ gây lỗi `np.stack`.")
        elif EXPECTED_EMBEDDING_DIM not in embedding_lengths and embedding_lengths: # có embedding nhưng không phải độ dài mong muốn
             logger.error(f"LỖI TIỀM ẨN: Độ dài embedding tìm thấy ({list(embedding_lengths.keys())[0]}) không khớp với EXPECTED_EMBEDDING_DIM ({EXPECTED_EMBEDDING_DIM}).")
        elif EXPECTED_EMBEDDING_DIM in embedding_lengths and len(embedding_lengths) == 1:
             logger.info(f"THÀNH CÔNG: Tất cả {embedding_lengths[EXPECTED_EMBEDDING_DIM]} embeddings hợp lệ đều có độ dài mong đợi là {EXPECTED_EMBEDDING_DIM}.")

    else:
        if total_detections_checked > 0 and (detections_missing_embedding_attr + detections_with_none_embedding == total_detections_checked):
            logger.error("LỖI TIỀM ẨN: Không tìm thấy embedding hợp lệ nào trong tất cả các detections đã kiểm tra.")
        elif total_detections_checked == 0:
             logger.info("Không có detection nào để kiểm tra.")
        else:
            logger.info("Không có embedding nào có độ dài hợp lệ được tìm thấy để thống kê (có thể tất cả đều None hoặc thiếu).")
    
    logger.info(f"--- Kết thúc kiểm tra tính nhất quán của embeddings ---\n")


def cluster_face_embeddings(dataset, patches_field="dlib_face_detections", embedding_attr_on_detection="custom_embedding", brain_key="dlib_face_clusters_viz"):
    """
    Chạy compute_visualization để phân cụm các embeddings khuôn mặt.
    """
    logger.info(f"Bắt đầu compute_visualization cho các detections trong trường: '{patches_field}'")
    # Khi patches_field được cung cấp, 'embeddings' nên là tên của thuộc tính embedding trên mỗi patch.
    logger.info(f"Sử dụng thuộc tính embedding: '{embedding_attr_on_detection}' trên mỗi detection trong '{patches_field}'.")
    
    results = fob.compute_visualization(
        dataset,
        patches_field=patches_field,
        embeddings=embedding_attr_on_detection, # SỬA Ở ĐÂY: Chỉ tên thuộc tính, không phải đường dẫn đầy đủ
        brain_key=brain_key,
        verbose=True,
        # Bạn có thể thêm các tham số khác cho method giảm chiều (ví dụ: umap, tsne, pca)
        # method="umap", 
        # num_dims=2, 
    )
    
    logger.info(f"Quá trình compute_visualization (bao gồm clustering tiềm năng) đã hoàn tất với brain_key: '{brain_key}'.")
    logger.info(f"Kiểm tra App hoặc dataset.get_brain_info('{brain_key}') để xem trạng thái và kết quả.")
    logger.info(f"Nhãn cluster (nếu được tạo) thường được thêm vào các đối tượng trong '{patches_field}.detections', ví dụ: trường '{brain_key}_cluster'.")

if __name__ == "__main__":
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    fo.config.database_name = "fiftyone" 
    logger.info(f"Kết nối tới MongoDB: {fo.config.database_uri}, DB: {fo.config.database_name}")

    dataset_name = "video_dataset_faces_dlib" 
    if not fo.dataset_exists(dataset_name):
        logger.error(f"Dataset '{dataset_name}' không tồn tại. Vui lòng chạy script face_recognition_utils.py trước.")
        exit()

    dataset = fo.load_dataset(dataset_name)
    logger.info(f"Đã tải dataset '{dataset_name}' với {len(dataset)} samples (frames).")

    # --- CẢNH BÁO ---
    # Việc xóa trường nên được thực hiện cẩn thận. 
    # Nếu bạn chắc chắn muốn làm mới hoàn toàn, hãy bỏ comment dòng này.
    target_detections_field = "dlib_face_detections" # Trường đích sẽ được tạo/ghi đè
    # if dataset.has_sample_field(target_detections_field):
    #     logger.info(f"Đang xóa trường '{target_detections_field}' cũ để làm mới...")
    #     dataset.delete_sample_field(target_detections_field)
    #     logger.info(f"Đã xóa trường '{target_detections_field}' cũ.")
    # --- HẾT CẢNH BÁO ---


    # Bước 1: Chuyển đổi cấu trúc dữ liệu
    logger.info("--- Bước 1: Bắt đầu chuyển đổi dlib_face_embeddings sang Detections ---")
    convert_dlib_embeddings_to_detections(dataset, 
                                          source_field="dlib_face_embeddings", 
                                          target_field=target_detections_field)
    logger.info(f"[✓] Đã hoàn tất chuyển đổi embeddings sang trường Detections '{target_detections_field}'")

    # Bước 2: Xác minh tính nhất quán của các embeddings vừa tạo
    logger.info("--- Bước 2: Bắt đầu xác minh tính nhất quán của custom_embedding ---")
    verify_embeddings_consistency(dataset, 
                                  detections_field=target_detections_field, 
                                  embedding_attr="custom_embedding")
    
    logger.info("Script hoàn tất. Hãy kiểm tra log để biết chi tiết về các vấn đề (nếu có).")
    logger.info("Nếu log cho thấy các embedding đã nhất quán, bạn có thể thử lại Clustering Plugin trong UI.")

    # Bước 3 (Tùy chọn từ script cũ): Chạy clustering bằng compute_visualization từ script
    # Nếu bạn muốn sử dụng clustering plugin từ UI, bạn có thể không cần chạy bước này từ script.
    # logger.info("--- Bước 3 (Tùy chọn): Chạy compute_visualization (có thể bao gồm clustering) ---")
    # cluster_face_embeddings(dataset, 
    #                         patches_field=target_detections_field, 
    #                         embedding_attr_on_detection="custom_embedding", 
    #                         brain_key="dlib_character_clusters_script_v2") # Sử dụng brain_key mới
    # logger.info("[✓] Quá trình compute_visualization (và clustering tiềm năng) từ script đã hoàn tất!")
    # logger.info("Khởi chạy FiftyOne App để khám phá kết quả nếu bạn chạy Bước 3.")

