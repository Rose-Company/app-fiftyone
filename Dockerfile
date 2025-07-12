FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libboost-all-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt
COPY scripts/requirements.txt /app/requirements.txt

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Cài đặt FiftyOne và tf-keras (cho DeepFace)
RUN pip install --no-cache-dir fiftyone tf-keras


# Sao chép toàn bộ script vào container
COPY scripts/ /app/scripts/

# Biến môi trường (đã có trong compose nên không bắt buộc)
#ENV FIFTYONE_DATABASE_URI=mongodb://mongo:27017
#ENV FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0

# Lệnh mặc định (bỏ qua, dùng `command:` trong compose)