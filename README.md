
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


import_videos -> 

# Video Character Search System

This system allows you to identify, search for, and navigate to scenes with specific characters in videos.

## Overview

The system processes videos through a pipeline:

1. **Import Videos**: Add videos to the FiftyOne dataset
2. **Extract Frames**: Sample frames from videos (1 per second)
3. **Detect People**: Find people in each frame
4. **Extract Faces**: Process detected people to find faces
5. **Group Faces**: Cluster similar faces to identify unique characters
6. **Character Search**: Create a searchable index and user interface

## Prerequisites

- Docker installed
- FiftyOne Docker container running
- Videos placed in the `/fiftyone/data/videos` directory

## Running the Pipeline

### Complete Pipeline

To run the entire pipeline from start to finish:

```bash
python /app/scripts/run_pipeline.py
```

This will process your videos through all steps and launch the FiftyOne app with character search functionality.

### Character Search Only

If you've already completed steps 1-5 and just want to use the character search interface:

```bash
python /app/scripts/run_character_search.py
```

## Using the Character Search Interface

1. Open the FiftyOne app at http://localhost:5151
2. Look for the "Character Search" panel in the sidebar
3. Use the interface to:
   - Select characters you want to find
   - Exclude characters you don't want in the scene
   - Set minimum scene duration
   - Click "Search for Character Appearances"
4. Review results and click on any scene to navigate directly to that point in the video

## Features

- **Character Detection**: Automatically identifies unique characters in the video
- **Temporal Indexing**: Maps when each character appears in the video
- **Flexible Search**: Find scenes based on which characters are present or absent
- **Direct Navigation**: Jump directly to scenes containing selected characters

## Individual Scripts

If needed, you can run individual steps:

- `import_videos.py`: Add videos to FiftyOne
- `extract_frames.py`: Sample frames from videos
- `detect_people.py`: Find people in frames
- `extract_faces.py`: Extract faces from person detections
- `group_faces_characters.py`: Group faces into character clusters
- `character_search.py`: Build the search index and launch the interface

## Troubleshooting

- If character grouping gives poor results, try adjusting clustering parameters in `group_faces_characters.py`
- For large videos, you may need to adjust batch sizes to prevent out-of-memory errors
- Check MongoDB connection if you see database connection errors

## Related Information

- FiftyOne Documentation: https://docs.voxel51.com/
- Face Recognition Documentation: https://github.com/ageitgey/face_recognition 