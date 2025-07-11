import fiftyone as fo
from fiftyone import ViewField as F
import os

# Cấu hình kết nối MongoDB trong container
fo.config.database_uri = "mongodb://mongo:27017"

# Load dataset video đã tạo trước đó
dataset_name = "video_dataset_final"
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
            video_id = os.path.splitext(video_name)[0]  # Bỏ extension (.mp4, .avi, etc.)
            
            # Lưu video_id vào sample
            sample["video_id"] = video_id
            sample.save()
            print(f"  Added video_id '{video_id}' to frame: {os.path.basename(sample.filepath)}")
    
    # Thêm frames mới vào dataset đã tồn tại
    frames_dataset.merge_samples(frames_view)
    
    print(f"\nĐã trích xuất thêm frames vào dataset '{frames_dataset_name}':")
    print(f"- Số lượng videos mới xử lý: {num_unprocessed}")
    print(f"- Số lượng frames mới đã trích xuất: {len(frames_view)}")

# Kiểm tra và cập nhật video_id cho các frames hiện có (nếu thiếu)
print(f"\nKiểm tra và cập nhật video_id cho các frames hiện có...")
updated_count = 0

for sample in frames_dataset:
    # Kiểm tra xem sample đã có video_id chưa
    if not sample.has_field("video_id") or sample.video_id is None:
        if hasattr(sample.metadata, "source_video_path"):
            # Lấy video_id từ source_video_path
            video_path = sample.metadata.source_video_path
            video_name = os.path.basename(video_path)
            video_id = os.path.splitext(video_name)[0]
            
            sample["video_id"] = video_id
            sample.save()
            updated_count += 1
        else:
            # Nếu không có source_video_path, thử lấy từ đường dẫn file
            filepath = sample.filepath
            # Đường dẫn thường có dạng: /fiftyone/data/frames/video_name/frame.jpg
            path_parts = filepath.split('/')
            if "frames" in path_parts:
                frames_index = path_parts.index("frames")
                if frames_index + 1 < len(path_parts):
                    video_id = path_parts[frames_index + 1]
                    sample["video_id"] = video_id
                    sample.save()
                    updated_count += 1

if updated_count > 0:
    print(f"Đã cập nhật video_id cho {updated_count} frames")
else:
    print("Tất cả frames đã có video_id")

# Hiển thị thông tin tổng hợp
print(f"\nTổng hợp dataset frames '{frames_dataset_name}':")
print(f"- Tổng số frames: {len(frames_dataset)}")

# Hiển thị thống kê theo video_id
video_id_stats = {}
for sample in frames_dataset:
    if sample.has_field("video_id") and sample.video_id:
        video_id = sample.video_id
        if video_id not in video_id_stats:
            video_id_stats[video_id] = 0
        video_id_stats[video_id] += 1

if video_id_stats:
    print(f"\nThống kê frames theo video_id:")
    for video_id, count in sorted(video_id_stats.items()):
        print(f"- {video_id}: {count} frames")

# Hiển thị thông tin chi tiết về frames
print(f"\nThông tin chi tiết một số frames:")
for sample in frames_dataset.take(5):  # Chỉ hiển thị 5 frames đầu tiên để demo
    metadata = sample.metadata
    if metadata:
        print(f"\nFrame: {os.path.basename(sample.filepath)}")
        if sample.has_field("video_id"):
            print(f"- Video ID: {sample.video_id}")
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