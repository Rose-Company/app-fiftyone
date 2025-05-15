import fiftyone as fo
from fiftyone import ViewField as F
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

# Tìm các video đã được xử lý trước đó
frames_dataset_name = f"{dataset_name}_frames"
processed_videos = set()

if fo.dataset_exists(frames_dataset_name):
    frames_dataset = fo.load_dataset(frames_dataset_name)
    print(f"Dataset frames '{frames_dataset_name}' đã tồn tại với {len(frames_dataset)} frames")
    
    # Lấy danh sách video đã xử lý rồi (từ metadata của frames)
    for sample in frames_dataset:
        if hasattr(sample.metadata, "source_video_path"):
            processed_videos.add(sample.metadata.source_video_path)
else:
    print(f"Tạo dataset frames mới '{frames_dataset_name}'")
    frames_dataset = fo.Dataset(frames_dataset_name)

# Lọc các videos chưa được xử lý
unprocessed_videos = dataset
if processed_videos:
    unprocessed_videos = dataset.exclude(F("filepath").is_in(list(processed_videos)))
num_unprocessed = len(unprocessed_videos)

if num_unprocessed == 0:
    print("Không có video mới cần trích xuất frames.")
else:
    print(f"Tìm thấy {num_unprocessed} videos mới cần trích xuất frames:")
    for sample in unprocessed_videos:
        print(f"- {os.path.basename(sample.filepath)}")
    
    # Trích xuất frames từ videos mới
    frames_view = unprocessed_videos.to_frames(
        sample_frames=True,  # Tự động trích xuất frames
        fps=1,              # 1 frame mỗi giây
        output_dir="/fiftyone/data/frames",  # Thư mục lưu frames
        frames_patt="%06d.jpg",    # Format tên file frame

    )
    
    # Thêm thông tin video_id cho mỗi frame để dễ dàng phân loại nguồn trong các xử lý sau
    for sample in frames_view:
        if hasattr(sample.metadata, "source_video_path"):
            # Lấy video_id từ đường dẫn
            video_path = sample.metadata.source_video_path
            video_name = os.path.basename(video_path)
            video_id = os.path.splitext(video_name)[0]
            
            # Lưu video_id vào metadata
            sample["video_id"] = video_id
            sample.save()
    
    # Thêm frames mới vào dataset đã tồn tại
    frames_dataset.merge_samples(frames_view)
    
    print(f"\nĐã trích xuất thêm frames vào dataset '{frames_dataset_name}':")
    print(f"- Số lượng videos mới xử lý: {num_unprocessed}")
    print(f"- Số lượng frames mới đã trích xuất: {len(frames_view)}")

# Hiển thị thông tin tổng hợp
print(f"\nTổng hợp dataset frames '{frames_dataset_name}':")
print(f"- Tổng số frames: {len(frames_dataset)}")

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