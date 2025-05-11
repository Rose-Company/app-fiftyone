import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# Kết nối MongoDB
fo.config.database_uri = "mongodb://mongo:27017"

# Load dataset frames đã tạo
frames_dataset_name = "video_dataset_frames"
if not fo.dataset_exists(frames_dataset_name):
    print(f"Không tìm thấy dataset '{frames_dataset_name}'")
    exit(1)

frames = fo.load_dataset(frames_dataset_name)
print(f"Đã load dataset '{frames_dataset_name}' với {len(frames)} frames")

# Tải mô hình Faster R-CNN từ Model Zoo
print("\nĐang tải mô hình phát hiện người...")
detector = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")

# Áp dụng mô hình để phát hiện người
print("\nĐang thực hiện nhận diện người...")
frames.apply_model(
    detector,
    label_field="detections",    # Tên trường lưu kết quả
    confidence_thresh=0.5,       # Ngưỡng tin cậy
    classes=["person"]           # Chỉ phát hiện class "person"
)

# Tạo view chỉ chứa frames có người
person_frames = frames.match(
    F("detections.detections").filter(F("label") == "person").length() > 0
)

# Lưu kết quả vào dataset mới
person_dataset_name = "video_dataset_person_frames"
if fo.dataset_exists(person_dataset_name):
    fo.delete_dataset(person_dataset_name)
person_dataset = person_frames.clone(person_dataset_name)

# Thống kê kết quả
total_frames = len(frames)
frames_with_people = len(person_dataset)
print(f"\nKết quả nhận diện người:")
print(f"- Tổng số frames: {total_frames}")
print(f"- Số frames có người: {frames_with_people}")
print(f"- Tỷ lệ frames có người: {(frames_with_people/total_frames)*100:.2f}%")
