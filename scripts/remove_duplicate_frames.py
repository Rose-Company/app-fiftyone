import fiftyone as fo
from fiftyone import ViewField as F
import argparse
import os
import datetime
import gc

def get_video_id_from_sample(sample):
    """
    Trích xuất video_id từ sample theo nhiều cách khác nhau
    """
    # Phương pháp 1: Trực tiếp từ field video_id
    if sample.has_field("video_id"):
        return sample.video_id
    
    # Phương pháp 2: Từ filepath
    if hasattr(sample, "filepath"):
        filepath = sample.filepath
        
        # Tìm từ đường dẫn: .../frames/video_id/frame.jpg
        parts = filepath.split('/')
        for i, part in enumerate(parts):
            if part == "frames" and i < len(parts) - 1:
                return parts[i+1]
        
        # Tìm từ tên file nếu có format video_id_frame.jpg
        filename = os.path.basename(filepath)
        filename_no_ext = os.path.splitext(filename)[0]
        if '_' in filename_no_ext:
            parts = filename_no_ext.split('_')
            if len(parts) >= 2:
                return '_'.join(parts[:-1])  # Lấy tất cả trừ phần cuối (frame number)
        
        # Phương pháp 3: Từ đường dẫn có chứa video name
        # Ví dụ: /data/frames/test11/123.jpg
        if '/frames/' in filepath:
            after_frames = filepath.split('/frames/')[-1]
            video_part = after_frames.split('/')[0]
            if video_part:
                return video_part
    
    return None

def get_frame_number_from_sample(sample):
    """
    Trích xuất frame_number từ sample
    """
    # Phương pháp 1: Trực tiếp từ field frame_number
    if sample.has_field("frame_number"):
        return sample.frame_number
    
    # Phương pháp 2: Từ filepath
    if hasattr(sample, "filepath"):
        filename = os.path.basename(sample.filepath)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Nếu tên file chỉ là số
        try:
            return int(filename_no_ext)
        except ValueError:
            pass
        
        # Nếu tên file có format video_id_frame
        if '_' in filename_no_ext:
            parts = filename_no_ext.split('_')
            try:
                return int(parts[-1])  # Lấy phần cuối
            except ValueError:
                pass
    
    return None

def list_videos_in_dataset(dataset_name="video_dataset_faces_deepface_arcface_retinaface_final"):
    """
    Liệt kê tất cả video ID trong dataset và số lượng frames của mỗi video
    """
    # Kết nối MongoDB
    database_uri = os.getenv("FIFTYONE_DATABASE_URI")
    if not database_uri:
        print("Error: FIFTYONE_DATABASE_URI environment variable is required!")
        return {}
    fo.config.database_uri = database_uri
    
    if not fo.dataset_exists(dataset_name):
        print(f"Không tìm thấy dataset '{dataset_name}'")
        return {}
    
    dataset = fo.load_dataset(dataset_name)
    
    # Thu thập thông tin video
    video_stats = {}
    for sample in dataset:
        video_id = get_video_id_from_sample(sample)
        if video_id:
            if video_id not in video_stats:
                video_stats[video_id] = 0
            video_stats[video_id] += 1
    
    return video_stats

def remove_duplicate_frames(dataset_name="video_dataset_faces_deepface_arcface_retinaface_final", video_id=None, backup=False, method="video_frame"):
    """
    Xóa các frames trùng lặp trong dataset
    
    Args:
        dataset_name: Tên của dataset cần xử lý
        video_id: ID của video cần xử lý (None = xử lý tất cả)
        backup: Nếu True, sẽ tạo bản sao lưu trước khi xóa
        method: Phương pháp xác định trùng lặp ("filepath", "video_frame", "all")
    """
    # Kết nối MongoDB
    database_uri = os.getenv("FIFTYONE_DATABASE_URI")
    if not database_uri:
        print("Error: FIFTYONE_DATABASE_URI environment variable is required!")
        return False
    fo.config.database_uri = database_uri
    
    # Kiểm tra dataset tồn tại
    if not fo.dataset_exists(dataset_name):
        print(f"Không tìm thấy dataset '{dataset_name}'")
        return False
    
    # Load dataset
    dataset = fo.load_dataset(dataset_name)
    total_frames = len(dataset)
    print(f"Đã load dataset '{dataset_name}' với {total_frames} frames")
    
    # Lọc theo video_id nếu được chỉ định
    if video_id:
        print(f"Đang lọc frames cho video ID: {video_id}")
        filtered_samples = []
        
        for sample in dataset:
            sample_video_id = get_video_id_from_sample(sample)
            if sample_video_id == video_id:
                filtered_samples.append(sample)
        
        if not filtered_samples:
            print(f"Không tìm thấy frames nào cho video ID '{video_id}'")
            return False
        
        print(f"Tìm thấy {len(filtered_samples)} frames cho video ID '{video_id}'")
        # Tạo view chỉ chứa samples của video này
        sample_ids = [sample.id for sample in filtered_samples]
        dataset_view = dataset.select(sample_ids)
    else:
        dataset_view = dataset
        print("Xử lý tất cả frames trong dataset")
    
    # Tạo bản sao lưu nếu cần
    if backup:
        backup_name = f"{dataset_name}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if video_id:
            backup_name += f"_{video_id}"
        print(f"Đang tạo bản sao lưu '{backup_name}'...")
        dataset.clone(backup_name)
        print(f"Đã tạo bản sao lưu thành công")
    
    # Xác định frames trùng lặp
    print(f"Đang xác định frames trùng lặp theo phương pháp: {method}...")
    
    duplicates = []
    
    if method == "filepath" or method == "all":
        # Tìm frames trùng lặp theo filepath
        print("Tìm frames trùng lặp theo filepath...")
        filepath_groups = {}
        
        for sample in dataset_view:
            filepath = sample.filepath
            if filepath not in filepath_groups:
                filepath_groups[filepath] = []
            filepath_groups[filepath].append(sample.id)
        
        for filepath, sample_ids in filepath_groups.items():
            if len(sample_ids) > 1:
                # Giữ lại frame đầu tiên, đánh dấu các frame còn lại để xóa
                duplicates.extend(sample_ids[1:])
                print(f"Phát hiện {len(sample_ids)} frames trùng lặp tại {filepath}")
    
    if method == "video_frame" or method == "all":
        # Tìm frames trùng lặp theo video_id và frame_number
        print("Tìm frames trùng lặp theo video_id và frame_number...")
        video_frame_groups = {}
        
        for sample in dataset_view:
            sample_video_id = get_video_id_from_sample(sample)
            frame_number = get_frame_number_from_sample(sample)
            
            if sample_video_id is not None and frame_number is not None:
                key = f"{sample_video_id}_{frame_number}"
                if key not in video_frame_groups:
                    video_frame_groups[key] = []
                video_frame_groups[key].append(sample.id)
        
        for key, sample_ids in video_frame_groups.items():
            if len(sample_ids) > 1:
                # Giữ lại frame đầu tiên, đánh dấu các frame còn lại để xóa
                for idx in range(1, len(sample_ids)):
                    if sample_ids[idx] not in duplicates:
                        duplicates.append(sample_ids[idx])
                video_id_key, frame_number = key.split('_')
                print(f"Phát hiện {len(sample_ids)} frames trùng lặp cho video {video_id_key}, frame {frame_number}")
    
    # Xóa frames trùng lặp
    num_duplicates = len(duplicates)
    print(f"Phát hiện tổng cộng {num_duplicates} frames trùng lặp")
    
    if num_duplicates > 0:
        confirm = input(f"Bạn có chắc chắn muốn xóa {num_duplicates} frames trùng lặp? (y/n): ")
        if confirm.lower() != 'y':
            print("Đã hủy quá trình xóa")
            return False
        
        print(f"Đang xóa {num_duplicates} frames trùng lặp...")
        dataset.delete_samples(duplicates)
        
        # Làm sạch bộ nhớ
        gc.collect()
        
        # Kiểm tra lại số lượng frames
        remaining_frames = len(dataset)
        print(f"Đã xóa thành công. Dataset còn lại {remaining_frames} frames (giảm {total_frames - remaining_frames} frames)")
    else:
        print("Không phát hiện frames trùng lặp")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xóa frames trùng lặp từ dataset")
    parser.add_argument("--dataset", default="video_dataset_faces_deepface_arcface_retinaface_final", help="Tên dataset cần xử lý")
    parser.add_argument("--video-id", help="ID của video cần xử lý (bỏ trống để xử lý tất cả)")
    parser.add_argument("--list-videos", action="store_true", help="Liệt kê tất cả video ID trong dataset")
    parser.add_argument("--backup", action="store_true", help="Tạo bản sao lưu trước khi xóa")
    parser.add_argument("--method", choices=["filepath", "video_frame", "all"], default="video_frame", 
                        help="Phương pháp xác định trùng lặp")
    parser.add_argument("--force", action="store_true", help="Không hỏi xác nhận khi xóa")
    
    args = parser.parse_args()
    
    # Liệt kê video nếu được yêu cầu
    if args.list_videos:
        print("Đang liệt kê tất cả video trong dataset...")
        video_stats = list_videos_in_dataset(args.dataset)
        if video_stats:
            print(f"\nTìm thấy {len(video_stats)} video trong dataset '{args.dataset}':")
            for video_id, frame_count in sorted(video_stats.items()):
                print(f"  - {video_id}: {frame_count} frames")
        else:
            print("Không tìm thấy video nào trong dataset")
        exit(0)
    
    # Ghi đè hàm input nếu force=True
    if args.force:
        def force_yes(prompt):
            print(prompt + " y (forced)")
            return "y"
        
        original_input = input
        input = force_yes
    
    try:
        remove_duplicate_frames(args.dataset, args.video_id, args.backup, args.method)
    finally:
        # Khôi phục input gốc nếu đã ghi đè
        if args.force:
            input = original_input 