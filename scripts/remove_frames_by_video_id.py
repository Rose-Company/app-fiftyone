import fiftyone as fo
from fiftyone import ViewField as F
import os
import shutil

# Cấu hình kết nối MongoDB trong container
database_uri = os.getenv("FIFTYONE_DATABASE_URI")
if not database_uri:
    print("Error: FIFTYONE_DATABASE_URI environment variable is required!")
    exit(1)
fo.config.database_uri = database_uri

def remove_frames_by_video_id(video_id, frames_dataset_name="video_dataset_faces_deepface_arcface_retinaface_final", remove_files=False):
    """
    Xóa tất cả frames của một video_id cụ thể
    
    Args:
        video_id (str): ID của video cần xóa frames (VD: "test11")
        frames_dataset_name (str): Tên dataset chứa frames
        remove_files (bool): Có xóa file ảnh trên disk hay không
    
    Returns:
        int: Số lượng frames đã xóa
    """
    
    if not fo.dataset_exists(frames_dataset_name):
        print(f"Không tìm thấy dataset '{frames_dataset_name}'")
        return 0
    
    # Load dataset frames
    frames_dataset = fo.load_dataset(frames_dataset_name)
    print(f"Đã load dataset '{frames_dataset_name}' với {len(frames_dataset)} frames")
    
    # Tìm tất cả frames của video_id cần xóa
    frames_to_remove = frames_dataset.match(F("video_id") == video_id)
    num_frames_to_remove = len(frames_to_remove)
    
    if num_frames_to_remove == 0:
        print(f"Không tìm thấy frames nào có video_id = '{video_id}'")
        return 0
    
    print(f"Tìm thấy {num_frames_to_remove} frames có video_id = '{video_id}'")
    
    # Thu thập danh sách file paths trước khi xóa
    file_paths = []
    frame_dirs = set()
    
    for sample in frames_to_remove:
        file_path = sample.filepath
        file_paths.append(file_path)
        # Lấy thư mục chứa frame để xóa sau
        frame_dir = os.path.dirname(file_path)
        frame_dirs.add(frame_dir)
        print(f"  - {os.path.basename(file_path)}")
    
    # Xác nhận xóa
    confirm = input(f"\nBạn có chắc chắn muốn xóa {num_frames_to_remove} frames của video '{video_id}'? (y/N): ")
    if confirm.lower() != 'y':
        print("Hủy bỏ thao tác xóa")
        return 0
    
    # Xóa samples khỏi dataset
    print(f"\nĐang xóa {num_frames_to_remove} samples khỏi dataset...")
    frames_dataset.delete_samples(frames_to_remove)
    
    # Xóa files trên disk nếu được yêu cầu
    if remove_files:
        print("Đang xóa files ảnh trên disk...")
        removed_files = 0
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed_files += 1
            except Exception as e:
                print(f"Lỗi khi xóa file {file_path}: {e}")
        
        print(f"Đã xóa {removed_files} files ảnh")
        
        # Xóa thư mục rỗng
        print("Đang kiểm tra và xóa thư mục rỗng...")
        for frame_dir in frame_dirs:
            try:
                if os.path.exists(frame_dir) and not os.listdir(frame_dir):
                    os.rmdir(frame_dir)
                    print(f"Đã xóa thư mục rỗng: {frame_dir}")
            except Exception as e:
                print(f"Lỗi khi xóa thư mục {frame_dir}: {e}")
    
    print(f"\nĐã xóa thành công {num_frames_to_remove} frames của video '{video_id}'")
    print(f"Dataset '{frames_dataset_name}' hiện có {len(frames_dataset)} frames")
    
    return num_frames_to_remove

def list_video_ids(frames_dataset_name="video_dataset_faces_deepface_arcface_retinaface_final"):
    """
    Hiển thị danh sách tất cả video_id có trong dataset frames
    """
    if not fo.dataset_exists(frames_dataset_name):
        print(f"Không tìm thấy dataset '{frames_dataset_name}'")
        return
    
    frames_dataset = fo.load_dataset(frames_dataset_name)
    
    # Thống kê frames theo video_id
    video_id_stats = {}
    for sample in frames_dataset:
        if sample.has_field("video_id") and sample.video_id:
            video_id = sample.video_id
            if video_id not in video_id_stats:
                video_id_stats[video_id] = 0
            video_id_stats[video_id] += 1
    
    print(f"\nDanh sách video_id trong dataset '{frames_dataset_name}':")
    print(f"Tổng số frames: {len(frames_dataset)}")
    print("─" * 50)
    for video_id, count in sorted(video_id_stats.items()):
        print(f"{video_id:15s}: {count:4d} frames")

if __name__ == "__main__":
    # ===========================================
    # HARDCODE VIDEO_ID CẦN XÓA TẠI ĐÂY
    # ===========================================
    VIDEO_ID_TO_REMOVE = "test11"  # Thay đổi video_id cần xóa tại đây
    # ===========================================
    
    # Hiển thị danh sách video_id hiện có
    list_video_ids()
    
    print(f"\nSẽ xóa frames của video_id: '{VIDEO_ID_TO_REMOVE}'")
    
    # Xóa frames
    removed_count = remove_frames_by_video_id(
        video_id=VIDEO_ID_TO_REMOVE,
        frames_dataset_name="video_dataset_faces_deepface_arcface_retinaface_final",
        remove_files=False  # CHỈ XÓA TRÊN DATASET, KHÔNG XÓA FILES TRÊN DISK
    )
    
    if removed_count > 0:
        print(f"\nThành công! Đã xóa {removed_count} frames của video '{VIDEO_ID_TO_REMOVE}'")
        # Hiển thị trạng thái sau khi xóa
        print("\nTrạng thái dataset sau khi xóa:")
        list_video_ids()
    else:
        print("Không có frames nào được xóa") 