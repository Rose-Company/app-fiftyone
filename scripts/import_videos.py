import fiftyone as fo
import os

# Cấu hình kết nối MongoDB từ biến môi trường
database_uri = os.getenv("FIFTYONE_DATABASE_URI")
if not database_uri:
    print("Error: FIFTYONE_DATABASE_URI environment variable is required!")
    exit(1)
fo.config.database_uri = database_uri

# Đường dẫn video từ biến môi trường
video_dir = os.getenv("VIDEO_DIR")
if not video_dir:
    print("Error: VIDEO_DIR environment variable is required!")
    exit(1)

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
dataset_name = "video_dataset_final"

# Load hoặc tạo dataset
if fo.dataset_exists(dataset_name):
    print(f"\nĐang sử dụng dataset '{dataset_name}' đã tồn tại")
    dataset = fo.load_dataset(dataset_name)
    
    # Lấy danh sách video đã tồn tại
    existing_videos = set(dataset.values("filepath"))
    print(f"Đã có {len(existing_videos)} videos trong dataset")
else:
    print(f"\nTạo dataset mới '{dataset_name}'")
    dataset = fo.Dataset(dataset_name)
    existing_videos = set()

# Lọc các video mới chưa được import
new_videos = [v for v in video_files if v not in existing_videos]

if not new_videos:
    print("\nKhông có video mới để import.")
else:
    print(f"\nImport {len(new_videos)} videos mới:")
    for v in new_videos:
        print(f"- {os.path.basename(v)}")

    # Import videos mới
    samples = []
    for video_path in new_videos:
        sample = fo.Sample(filepath=video_path)
        samples.append(sample)

    # Thêm samples vào dataset
    dataset.add_samples(samples)

    # Tính toán metadata cho samples mới
    for sample in samples:
        sample.compute_metadata()
        sample.save()

    print(f"\nĐã import thành công {len(new_videos)} videos mới vào dataset '{dataset.name}'")

print(f"Tổng số videos trong dataset: {len(dataset)}")
# Lấy port từ biến môi trường
fiftyone_port = os.getenv("FIFTYONE_PORT")
if not fiftyone_port:
    print("Error: FIFTYONE_PORT environment variable is required!")
    exit(1)
print(f"Truy cập http://localhost:{fiftyone_port} để xem dataset")

# In danh sách tất cả datasets
print("\nDanh sách tất cả datasets:")
for name in fo.list_datasets():
    ds = fo.load_dataset(name)
    print(f"- {name}: {len(ds)} samples")