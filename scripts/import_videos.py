import fiftyone as fo
import os

# Cấu hình kết nối MongoDB trong container
fo.config.database_uri = "mongodb://mongo:27017"

# Đường dẫn trong container (đã được mount)
video_dir = "/fiftyone/data/videos"

# Kiểm tra thư mục tồn tại
if not os.path.exists(video_dir):
    print(f"Thư mục {video_dir} không tồn tại!")
    exit(1)

# Liệt kê files trong thư mục
video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
               if f.endswith(".mp4")]

if not video_files:
    print(f"Không tìm thấy file MP4 nào trong {video_dir}")
    exit(1)

print(f"Tìm thấy {len(video_files)} file video:")
for video in video_files:
    print(f"- {os.path.basename(video)}")

# Tên dataset
dataset_name = "video_dataset"

# Load hoặc tạo dataset
if fo.dataset_exists(dataset_name):
    print(f"\nĐang sử dụng dataset '{dataset_name}' đã tồn tại")
    dataset = fo.load_dataset(dataset_name)
    # Xóa tất cả samples cũ (tùy chọn)
    dataset.clear()
else:
    print(f"\nTạo dataset mới '{dataset_name}'")
    dataset = fo.Dataset(dataset_name)

# Import videos
samples = []
for video_path in video_files:
    sample = fo.Sample(filepath=video_path)
    samples.append(sample)

# Thêm samples vào dataset
dataset.add_samples(samples)

# Tính toán metadata
dataset.compute_metadata()

print(f"\nĐã import thành công {len(dataset)} videos vào dataset '{dataset.name}'")
print("Truy cập http://localhost:5151 để xem dataset")

# In danh sách tất cả datasets
print("\nDanh sách tất cả datasets:")
for name in fo.list_datasets():
    ds = fo.load_dataset(name)
    print(f"- {name}: {len(ds)} samples")