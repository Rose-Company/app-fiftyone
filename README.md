
# FiftyOne Docker Project

Dự án này cung cấp một môi trường Docker hoàn chỉnh để chạy FiftyOne và xử lý video/ảnh với khả năng phát hiện người và nhận diện khuôn mặt.

## Cấu trúc thư mục
fiftyone-docker/
├── docker-compose.yml - File cấu hình Docker Compose
├── Dockerfile - File build image Docker
├── scripts/ - Thư mục chứa các script Python
└── data/ - Thư mục chứa dữ liệu (videos, frames)



## Cài đặt

### Yêu cầu

- Docker và Docker Compose
- Git

### Các bước cài đặt

1. Clone repository:
```bash
git clone https://github.com/[username]/fiftyone-docker.git
cd fiftyone-docker
```

2. Tạo các thư mục data:
```bash
mkdir -p data/videos data/frames
```

3. Khởi động container:
```bash
docker-compose up -d
```

4. Truy cập FiftyOne UI tại: http://localhost:5151

## Sử dụng

### Import video

```bash
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/import_videos.py
```

### Trích xuất frames từ video

```bash
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/extract_frames.py
```

### Phát hiện người trong frames

```bash
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/detect_people.py
```

### Nhận diện khuôn mặt

```bash
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/face_recognition_utils.py
```
### Gom nhóm khuôn mặt

```bash
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/group_faces_characters.py
```

## Quản lý dữ liệu

Dữ liệu được lưu trữ trong MongoDB và có thể xem qua:
- FiftyOne UI: http://localhost:5151
- MongoDB Compass: mongodb://localhost:27017

## Dừng và xóa container

```bash
# Dừng container
docker-compose down

# Xóa volume (mất dữ liệu)
docker-compose down -v
```
## Cài các thư viện nếu không muốn build lại Dockerfile
```bash
docker-compose exec fiftyone pip install hdbscan

```
## Test thử các video riêng biệt
```bash
## Chaỵ frame mới chưa được detect 
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/detect_people_by_video.py test2 
## Chạy lại tất cả frame của video 
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/detect_people_by_video.py test2 

# Tạo dataset mới và xử lý lại, không ghi đè dataset cũ
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/extract_faces_by_video.py test --reprocess --new-dataset

# Xử lý lại và ghi đè lên dataset cũ, không cần xác nhận
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/extract_faces_by_video.py test --reprocess --force

# Xoá trùng lặp trong dataset faces
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/remove_duplicate_frames.py --dataset video_dataset_faces


```
## Ghi chú

- Đặt video trong thư mục `data/videos/` để import
- Frames được trích xuất lưu trong `data/frames/`
- MongoDB lưu trữ metadata trong volume Docker







docker exec -it fiftyone-docker-fiftyone-1 fiftyone plugins download https://github.com/jacobmarks/clustering-plugin

docker exec -it fiftyone-docker-fiftyone-1 pip install umap-learn

