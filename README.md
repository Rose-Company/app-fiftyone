
# FiftyOne Docker - Video Character Analysis System

Hệ thống phân tích nhân vật trong video sử dụng FiftyOne, AI face recognition và Docker. Cho phép tự động phát hiện, nhận diện và tìm kiếm nhân vật trong video.

## 🎯 Tính năng chính

- **Phân tích video tự động**: Trích xuất frames, phát hiện người, nhận diện khuôn mặt
- **Nhận diện nhân vật**: Gom nhóm khuôn mặt thành các nhân vật duy nhất
- **Tìm kiếm thông minh**: Tìm cảnh theo nhân vật, thời gian, điều kiện
- **Giao diện trực quan**: FiftyOne UI để xem và phân tích dữ liệu
- **Quản lý môi trường**: Cấu hình linh hoạt qua file .env

## 📋 Yêu cầu hệ thống

- Docker và Docker Compose
- Git
- Tối thiểu 8GB RAM (khuyến nghị 16GB)
- 10GB dung lượng trống

## 🚀 Cài đặt nhanh

### 1. Clone repository
```bash
git clone <repository-url>
cd fiftyone-docker
```

### 2. Cấu hình môi trường
```bash
# Copy file cấu hình mẫu
cp .env.example .env

# Chỉnh sửa cấu hình nếu cần
nano .env
```

### 3. Tạo thư mục dữ liệu
```bash
mkdir -p data/videos data/frames
```

### 4. Khởi động hệ thống
```bash
# Build và khởi động container
docker-compose up -d

# Kiểm tra trạng thái
docker-compose ps
```

### 5. Truy cập giao diện
- **FiftyOne UI**: http://localhost:5151
- **MongoDB**: mongodb://localhost:27017

## 📁 Cấu trúc dự án

```
fiftyone-docker/
├── .env                     # Cấu hình môi trường (không commit)
├── .env.example            # Mẫu cấu hình
├── docker-compose.yml      # Cấu hình Docker Compose
├── Dockerfile             # Docker image definition
├── data/                  # Dữ liệu (được mount vào container)
│   ├── videos/           # Video nguồn
│   └── frames/           # Frames được trích xuất
└── scripts/              # Python scripts
    ├── import_videos.py          # Import video vào FiftyOne
    ├── extract_frames.py         # Trích xuất frames
    ├── extract_faces_deepface.py # Phát hiện khuôn mặt
    ├── group_faces_graph.py      # Gom nhóm khuôn mặt
    ├── character_search.py       # Tìm kiếm nhân vật
    ├── remove_frames_by_video_id.py # Xóa frames theo video
    └── test_env_vars.py          # Test cấu hình môi trường
```




## 📊 Quy trình xử lý

### 1. Import Videos
```bash
# Đặt video vào thư mục data/videos/
cp your_video.mp4 data/videos/

# Import vào FiftyOne
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/import_videos.py
```

### 2. Trích xuất Frames
```bash
# Trích xuất frames (1 frame/giây)
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/extract_frames.py
```

### 3. Phát hiện Khuôn mặt
```bash
# Sử dụng DeepFace + RetinaFace
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/extract_faces_deepface.py
```

### 4. Gom nhóm Nhân vật
```bash
# Clustering khuôn mặt thành nhân vật
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/group_faces_graph.py
```

### 5. Tạo Index Tìm kiếm
```bash
# Tạo character index
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/character_search.py
```

## 🔍 Tìm kiếm và Phân tích

### Character Search
- Tìm cảnh theo nhân vật cụ thể
- Loại trừ nhân vật không mong muốn
- Lọc theo thời gian xuất hiện
- Kết quả được lưu trong `character_index.json`

### Quản lý Dữ liệu
```bash
# Xóa frames của video cụ thể
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/remove_frames_by_video_id.py

# Test cấu hình môi trường
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/test_env_vars.py
```

## 🛠️ Các lệnh hữu ích

### Quản lý Container
```bash
# Khởi động
docker-compose up -d

# Dừng
docker-compose down

# Xem logs
docker-compose logs -f

# Rebuild image
docker-compose build --no-cache

# Vào container
docker exec -it fiftyone-docker-fiftyone-1 bash
```

### Cài đặt thêm thư viện
```bash
# Cài đặt trong container đang chạy
docker exec -it fiftyone-docker-fiftyone-1 pip install deepface

# Hoặc thêm vào Dockerfile và rebuild
```

### Sao lưu và Khôi phục
```bash
# Sao lưu MongoDB
docker exec fiftyone-docker-mongo-1 mongodump --out /data/db/backup

# Khôi phục
docker exec fiftyone-docker-mongo-1 mongorestore /data/db/backup
```

## 🔧 Troubleshooting

### Lỗi thường gặp

**1. Container không khởi động**
```bash
# Kiểm tra logs
docker-compose logs

# Kiểm tra port conflict
netstat -tulpn | grep 5151
```

**2. Lỗi môi trường**
```bash
# Test cấu hình
docker exec -it fiftyone-docker-fiftyone-1 python /app/scripts/test_env_vars.py

# Kiểm tra file .env
cat .env
```

**3. Lỗi memory**
```bash
# Tăng memory limit trong .env
MEMORY_LIMIT=32g

# Restart container
docker-compose down && docker-compose up -d
```

**4. Lỗi DeepFace/tf-keras**
```bash
# Cài đặt tf-keras
docker exec -it fiftyone-docker-fiftyone-1 pip install tf-keras

# Hoặc rebuild image
docker-compose build --no-cache
```

### Performance Tuning

**Tăng hiệu suất:**
- Tăng `MEMORY_LIMIT` và `SHARED_MEMORY_SIZE`
- Điều chỉnh `OMP_NUM_THREADS` và `PYTORCH_NUM_THREADS`
- Sử dụng SSD cho thư mục `data/`

**Giảm memory usage:**
- Xử lý video nhỏ hơn
- Giảm fps khi extract frames
- Batch processing với kích thước nhỏ hơn

## 📊 Datasets và Outputs

### FiftyOne Datasets
- `video_dataset_final`: Video gốc
- `video_dataset_final_frames`: Frames đã trích xuất
- `video_dataset_faces_deepface_arcface_retinaface_final`: Khuôn mặt đã phát hiện

### JSON Outputs
- `character_index.json`: Index tìm kiếm nhân vật
- `character_update.json`: Cập nhật gần đây
- `character_update_metadata.json`: Metadata thay đổi

## 🔐 Bảo mật

- File `.env` chứa thông tin nhạy cảm - **KHÔNG commit**
- Sử dụng `.env.example` làm template
- Cấu hình firewall cho production
- Thay đổi default ports nếu cần

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

[Thêm license information]

## 📞 Hỗ trợ

- **Issues**: Tạo issue trên GitHub
- **Documentation**: FiftyOne docs tại https://docs.voxel51.com/
- **Docker**: https://docs.docker.com/

---

**Lưu ý**: Đây là hệ thống phân tích video AI, cần tài nguyên tính toán đáng kể. Khuyến nghị chạy trên máy có GPU để tăng tốc độ xử lý. 