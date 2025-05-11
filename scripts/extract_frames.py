import fiftyone as fo
import os

# Cấu hình kết nối MongoDB trong container
fo.config.database_uri = "mongodb://mongo:27017"

# Load dataset video đã tạo trước đó
dataset_name = "video_dataset"
if not fo.dataset_exists(dataset_name):
    print(f"Không tìm thấy dataset '{dataset_name}'")
    exit(1)

dataset = fo.load_dataset(dataset_name)
print(f"Đã load dataset '{dataset_name}' với {len(dataset)} videos")

# Tạo dataset mới để lưu frames
frames_dataset_name = f"{dataset_name}_frames"
if fo.dataset_exists(frames_dataset_name):
    print(f"Xóa dataset frames cũ '{frames_dataset_name}'")
    fo.delete_dataset(frames_dataset_name)

# Trích xuất frames từ videos
frames_view = dataset.to_frames(
    sample_frames=True,  # Tự động trích xuất frames
    fps=1,              # 1 frame mỗi giây
    output_dir="/fiftyone/data/frames",  # Thư mục lưu frames
    frames_patt="%06d.jpg"    # Format tên file frame
)

# Lưu frames view thành dataset mới
frames_dataset = frames_view.clone(frames_dataset_name)

print(f"\nĐã trích xuất frames vào dataset '{frames_dataset_name}':")
print(f"- Số lượng videos gốc: {len(dataset)}")
print(f"- Số lượng frames đã trích xuất: {len(frames_dataset)}")

# Hiển thị thông tin chi tiết về frames
for sample in frames_dataset.take(5):  # Chỉ hiển thị 5 frames đầu tiên để demo
    metadata = sample.metadata
    if metadata:
        print(f"\nFrame từ video: {os.path.basename(sample.filepath)}")
        print(f"- Kích thước: {metadata.width}x{metadata.height}")
        if hasattr(sample, "frame_number"):
            print(f"- Frame số: {sample.frame_number}")
        if hasattr(sample, "timestamp"):
            print(f"- Timestamp: {sample.timestamp:.2f}s")

print("\nTruy cập http://localhost:5151 để xem frames đã trích xuất")

# In danh sách tất cả datasets
print("\nDanh sách tất cả datasets:")
for name in fo.list_datasets():
    ds = fo.load_dataset(name)
    print(f"- {name}: {len(ds)} samples")