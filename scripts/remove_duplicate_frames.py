import fiftyone as fo
from fiftyone import ViewField as F
import argparse
import os
import datetime
import gc

def remove_duplicate_frames(dataset_name="video_dataset_faces_dic", backup=True, method="filepath"):
    """
    Xóa các frames trùng lặp trong dataset, chỉ giữ lại một frame duy nhất
    
    Args:
        dataset_name: Tên của dataset cần xử lý
        backup: Nếu True, sẽ tạo bản sao lưu trước khi xóa
        method: Phương pháp xác định trùng lặp ("filepath", "video_frame", "all")
    """
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Kiểm tra dataset tồn tại
    if not fo.dataset_exists(dataset_name):
        print(f"Không tìm thấy dataset '{dataset_name}'")
        return False
    
    # Load dataset
    dataset = fo.load_dataset(dataset_name)
    total_frames = len(dataset)
    print(f"Đã load dataset '{dataset_name}' với {total_frames} frames")
    
    # Tạo bản sao lưu nếu cần
    if backup:
        backup_name = f"{dataset_name}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Đang tạo bản sao lưu '{backup_name}'...")
        dataset.clone(backup_name)
        print(f"Đã tạo bản sao lưu thành công")
    
    # Xác định frames trùng lặp
    print(f"Đang xác định frames trùng lặp theo phương pháp: {method}...")
    
    duplicates = []
    frames_to_keep = []
    
    if method == "filepath" or method == "all":
        # Tìm frames trùng lặp theo filepath
        print("Tìm frames trùng lặp theo filepath...")
        filepath_groups = {}
        
        # Thu thập frames theo filepath
        for sample in dataset:
            filepath = sample.filepath
            if filepath not in filepath_groups:
                filepath_groups[filepath] = []
            filepath_groups[filepath].append(sample.id)
        
        # Xác định frames trùng lặp
        for filepath, sample_ids in filepath_groups.items():
            if len(sample_ids) > 1:
                # Giữ lại frame đầu tiên, đánh dấu các frame còn lại để xóa
                frames_to_keep.append(sample_ids[0])
                duplicates.extend(sample_ids[1:])
                print(f"Phát hiện {len(sample_ids)} frames trùng lặp tại {filepath}")
    
    if method == "video_frame" or method == "all":
        # Tìm frames trùng lặp theo video_id và frame_number
        print("Tìm frames trùng lặp theo video_id và frame_number...")
        video_frame_groups = {}
        
        # Thu thập frames theo video_id và frame_number
        for sample in dataset:
            video_id = None
            frame_number = None
            
            # Lấy video_id từ các thuộc tính khác nhau
            if sample.has_field("video_id"):
                video_id = sample.video_id
            elif hasattr(sample, "filepath"):
                # Tìm từ filepath
                parts = sample.filepath.split('/')
                for i, part in enumerate(parts):
                    if part == "frames" and i < len(parts) - 1:
                        video_id = parts[i+1]
                        break
            
            # Lấy frame_number
            if sample.has_field("frame_number"):
                frame_number = sample.frame_number
            elif hasattr(sample, "filepath"):
                # Trích xuất từ tên file
                filename = os.path.basename(sample.filepath)
                filename_no_ext = os.path.splitext(filename)[0]
                try:
                    frame_number = int(filename_no_ext)
                except ValueError:
                    # Không thể chuyển thành số
                    continue
            
            if video_id is not None and frame_number is not None:
                key = f"{video_id}_{frame_number}"
                if key not in video_frame_groups:
                    video_frame_groups[key] = []
                video_frame_groups[key].append(sample.id)
        
        # Xác định frames trùng lặp
        for key, sample_ids in video_frame_groups.items():
            if len(sample_ids) > 1:
                # Nếu frame đầu tiên chưa được đánh dấu xóa từ phương pháp filepath
                if sample_ids[0] not in duplicates and sample_ids[0] not in frames_to_keep:
                    frames_to_keep.append(sample_ids[0])
                # Đánh dấu các frame còn lại để xóa nếu chưa được đánh dấu
                for idx in range(1, len(sample_ids)):
                    if sample_ids[idx] not in duplicates:
                        duplicates.append(sample_ids[idx])
                video_id, frame_number = key.split('_')
                print(f"Phát hiện {len(sample_ids)} frames trùng lặp cho video {video_id}, frame {frame_number}")
    
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
    parser.add_argument("--dataset", default="video_dataset_faces_dic", help="Tên dataset cần xử lý")
    parser.add_argument("--no-backup", action="store_true", help="Không tạo bản sao lưu trước khi xóa")
    parser.add_argument("--method", choices=["filepath", "video_frame", "all"], default="all", 
                        help="Phương pháp xác định trùng lặp: 'filepath', 'video_frame', hoặc 'all'")
    parser.add_argument("--force", action="store_true", help="Không hỏi xác nhận khi xóa")
    
    args = parser.parse_args()
    
    # Ghi đè hàm input nếu force=True
    if args.force:
        def force_yes(prompt):
            print(prompt + " y (forced)")
            return "y"
        
        # Lưu input gốc
        original_input = input
        # Ghi đè input
        input = force_yes
    
    try:
        remove_duplicate_frames(args.dataset, not args.no_backup, args.method)
    finally:
        # Khôi phục input gốc nếu đã ghi đè
        if args.force:
            input = original_input 