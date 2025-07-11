import fiftyone as fo

DATASET_NAME = "video_dataset_faces_dlib_test_cnn_17/6"
VIDEO_ID_VALUE = "image_dataset_17/6"  # Hoặc bạn có thể lấy theo tên thư mục cha

def add_video_id_field(dataset_name, video_id_value):
    fo.config.database_uri = "mongodb://mongo:27017"
    dataset = fo.load_dataset(dataset_name)
    print(f"Loaded dataset: {dataset_name} ({len(dataset)} samples)")

    # Thêm trường video_id cho tất cả sample
    for sample in dataset.iter_samples(autosave=True):
        # Nếu muốn lấy tên thư mục cha làm video_id:
        # import os
        # video_id = os.path.basename(os.path.dirname(sample.filepath))
        # sample["video_id"] = video_id

        # Nếu muốn gán cùng một giá trị:
        sample["video_id"] = video_id_value

    print("Đã thêm trường video_id cho toàn bộ dataset.")

if __name__ == "__main__":
    add_video_id_field(DATASET_NAME, VIDEO_ID_VALUE)