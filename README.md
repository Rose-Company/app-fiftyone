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

## Ghi chú

- Đặt video trong thư mục `data/videos/` để import
- Frames được trích xuất lưu trong `data/frames/`
- MongoDB lưu trữ metadata trong volume Docker