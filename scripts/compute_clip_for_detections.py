import fiftyone as fo
import fiftyone.zoo as foz
import logging
from fiftyone import ViewField as F

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_clip_embeddings_to_detections(
    dataset_name: str,
    patches_field: str,
    clip_embedding_field: str = "clip_embedding",
    clip_model_name: str = "clip-vit-base32-torch",
    batch_size: int = 8  # Điều chỉnh batch_size dựa trên bộ nhớ GPU/CPU của bạn
):
    """
    Tính toán và thêm CLIP embeddings vào các detections trong một dataset.

    Args:
        dataset_name (str): Tên của dataset FiftyOne.
        patches_field (str): Tên trường ở cấp sample chứa các đối tượng Detections
                             (ví dụ: "dlib_face_detections", "person_detections").
                             Mỗi detection trong list này sẽ được tính embedding.
        clip_embedding_field (str, optional): Tên trường mới sẽ được tạo bên trong mỗi
                                            Detection object để lưu trữ CLIP embedding.
                                            Mặc định là "clip_embedding".
        clip_model_name (str, optional): Tên của model CLIP từ FiftyOne Model Zoo.
                                        Mặc định là "clip-vit-base32-torch".
        batch_size (int, optional): Kích thước batch để xử lý embeddings.
    """
    logger.info(f"--- Bắt đầu tính toán CLIP embeddings cho dataset: {dataset_name} ---")

    # 1. Tải Dataset
    if not fo.dataset_exists(dataset_name):
        logger.error(f"Lỗi: Dataset '{dataset_name}' không tồn tại.")
        return
    dataset = fo.load_dataset(dataset_name)
    logger.info(f"Đã tải dataset '{dataset_name}' với {len(dataset)} samples.")

    # Kiểm tra xem patches_field có tồn tại không
    if not dataset.has_sample_field(patches_field):
        logger.error(f"Lỗi: Dataset '{dataset_name}' không có trường patches '{patches_field}'.")
        # In ra các trường có sẵn để người dùng tham khảo
        logger.info(f"Các trường sample có sẵn: {list(dataset.get_field_schema().keys())}")
        return

    # 2. Tải Model CLIP
    logger.info(f"Đang tải model CLIP: {clip_model_name}...")
    try:
        clip_model = foz.load_zoo_model(clip_model_name)
        logger.info("Tải model CLIP thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi tải model CLIP '{clip_model_name}': {e}")
        return

    # 3. Tính toán và Lưu trữ Embeddings
    # FiftyOne cung cấp một cách tiện lợi để tính embeddings cho các patches (detections)
    # Phương thức này sẽ tự động:
    #   - Trích xuất các vùng ảnh (patches) từ mỗi detection trong `patches_field`.
    #   - Đưa các patches qua model CLIP.
    #   - Lưu trữ embedding kết quả vào một trường mới (`clip_embedding_field`)
    #     BÊN TRONG mỗi object Detection tương ứng.

    logger.info(f"Đang tính toán CLIP embeddings cho các detections trong '{patches_field}' và lưu vào '{clip_embedding_field}'...")
    logger.info(f"Sử dụng batch size: {batch_size}")

    try:
        # dataset.compute_patch_embeddings() sẽ xử lý việc lặp qua các sample và detections
        # và lưu trữ embedding vào từng detection.
        dataset.compute_patch_embeddings(
            model=clip_model,
            patches_field=patches_field, # Trường chứa list Detections ở cấp sample
            embeddings_field=clip_embedding_field, # Tên trường mới cho embedding trên mỗi Detection
            batch_size=batch_size
        )
        logger.info("Tính toán và lưu trữ CLIP embeddings hoàn tất.")

        # Xác minh (tùy chọn, có thể mất thời gian với dataset lớn)
        logger.info("Đang xác minh một vài embedding đã được thêm...")
        # Lấy sample đầu tiên CÓ detections trong patches_field
        # và các detections đó CÓ trường clip_embedding_field vừa tạo
        # Điều này đảm bảo chúng ta đang kiểm tra một detection thực sự đã được xử lý
        view_with_clip_embeddings = dataset.match(
            F(f"{patches_field}.detections.{clip_embedding_field}").exists()
        )
        sample_with_processed_detection = view_with_clip_embeddings.first()

        if sample_with_processed_detection:
            detections_obj = sample_with_processed_detection[patches_field]
            first_detection_with_clip = None
            for det in detections_obj.detections:
                if hasattr(det, clip_embedding_field) and getattr(det, clip_embedding_field) is not None:
                    first_detection_with_clip = det
                    break
            
            if first_detection_with_clip:
                emb = getattr(first_detection_with_clip, clip_embedding_field)
                logger.info(f"Xác minh thành công! Sample ID: {sample_with_processed_detection.id}, "
                            f"Detection ID: {first_detection_with_clip.id} (Label: {first_detection_with_clip.label}) "
                            f"có '{clip_embedding_field}' với độ dài: {len(emb)} (Thường là 512 cho ViT-B/32).")
            else:
                # Điều này không nên xảy ra nếu sample_with_processed_detection được tìm thấy đúng cách
                logger.warning(f"Không tìm thấy detection nào có trường '{clip_embedding_field}' trong sample {sample_with_processed_detection.id}, dù sample này khớp với query.")
        else:
            logger.warning(f"Không tìm thấy sample nào có detections chứa trường '{clip_embedding_field}' để xác minh. "
                           f"Có thể không có detections nào được xử lý, hoặc trường patches_field trống, hoặc có lỗi trong quá trình tính toán.")

    except Exception as e:
        logger.error(f"Lỗi trong quá trình tính toán patch embeddings: {e}", exc_info=True)

    logger.info(f"--- Hoàn tất xử lý CLIP embeddings cho dataset: {dataset_name} ---")


if __name__ == "__main__":
    # --- Cấu hình --- 
    # Kết nối MongoDB (nhất quán với các file khác của bạn)
    fo.config.database_uri = "mongodb://mongo:27017" # Ví dụ cho Docker
    fo.config.database_name = "fiftyone" # Tên database mặc định hoặc của bạn
    logger.info(f"Kết nối tới MongoDB: {fo.config.database_uri}, DB: {fo.config.database_name}")

    TARGET_DATASET_NAME = "video_dataset_faces_test" # THAY THẾ bằng tên dataset của bạn
    
    # Trường ở cấp sample chứa các Detections objects.
    # Ví dụ: 'dlib_face_detections' (nếu bạn đã chuyển đổi từ dlib_face_embeddings)
    # hoặc 'person_detections' (nếu từ một model phát hiện người như YOLO).
    # Đây PHẢI LÀ tên của trường chứa một đối tượng fiftyone.core.labels.Detections.
    DETECTIONS_FIELD_ON_SAMPLE = "extracted_face_detections" # THAY THẾ bằng tên trường detections của bạn

    # Tên trường mới bạn muốn tạo bên trong mỗi Detection object để lưu CLIP embedding
    NEW_CLIP_EMBEDDING_FIELD_ON_DETECTION = "clip_embedding_vitb32"
    
    # Tên model CLIP từ Zoo
    CLIP_MODEL = "clip-vit-base32-torch" # Các lựa chọn khác: "clip-vit-large-patch14-torch", ...
    
    PROCESSING_BATCH_SIZE = 16 # Giảm nếu gặp lỗi OOM (Out of Memory)
    # --- Kết thúc cấu hình --- 

    logger.info(f"Dataset mục tiêu: {TARGET_DATASET_NAME}")
    logger.info(f"Trường Detections (cấp sample) sẽ xử lý: {DETECTIONS_FIELD_ON_SAMPLE}")
    logger.info(f"Trường mới cho CLIP embedding (trên mỗi detection): {NEW_CLIP_EMBEDDING_FIELD_ON_DETECTION}")
    logger.info(f"Model CLIP sẽ sử dụng: {CLIP_MODEL}")

    add_clip_embeddings_to_detections(
        dataset_name=TARGET_DATASET_NAME,
        patches_field=DETECTIONS_FIELD_ON_SAMPLE,
        clip_embedding_field=NEW_CLIP_EMBEDDING_FIELD_ON_DETECTION,
        clip_model_name=CLIP_MODEL,
        batch_size=PROCESSING_BATCH_SIZE
    )

    logger.info("\nHoàn thành! Bạn có thể mở FiftyOne App để xem kết quả.")
    logger.info(f"Trong FiftyOne App, mỗi detection trong trường '{DETECTIONS_FIELD_ON_SAMPLE}' "
                f"bây giờ nên có một trường con tên là '{NEW_CLIP_EMBEDDING_FIELD_ON_DETECTION}'.")
    logger.info("Bạn có thể sử dụng các embedding này để tìm kiếm tương đồng, gom cụm, v.v...")

    # Ví dụ cách xem một sample sau khi chạy (đã được cải thiện để dùng F operator):
    if fo.dataset_exists(TARGET_DATASET_NAME):
        dataset = fo.load_dataset(TARGET_DATASET_NAME)
        # Sử dụng F object để query chính xác hơn
        query_field = f"{DETECTIONS_FIELD_ON_SAMPLE}.detections.{NEW_CLIP_EMBEDDING_FIELD_ON_DETECTION}"
        sample_with_clip = dataset.match(F(query_field).exists()).first()
        
        if sample_with_clip:
            logger.info("\nThông tin sample đầu tiên có CLIP embedding:")
            logger.info(f"Sample ID: {sample_with_clip.id}")
            detections_obj = sample_with_clip[DETECTIONS_FIELD_ON_SAMPLE]
            if detections_obj and detections_obj.detections: # Kiểm tra lại cho chắc chắn
                for i, det in enumerate(detections_obj.detections):
                    if hasattr(det, NEW_CLIP_EMBEDDING_FIELD_ON_DETECTION) and getattr(det, NEW_CLIP_EMBEDDING_FIELD_ON_DETECTION) is not None:
                        logger.info(f"  Detection {i} - Label: {det.label}, BBox: {det.bounding_box}, "
                                    f"CLIP Embedding (độ dài): {len(getattr(det, NEW_CLIP_EMBEDDING_FIELD_ON_DETECTION))}")
            else:
                logger.info(f"Sample {sample_with_clip.id} không có detections trong trường '{DETECTIONS_FIELD_ON_SAMPLE}'.")
        else:
            logger.info(f"\nKhông tìm thấy sample nào có trường '{query_field}' tồn tại trong các detections của nó.")
