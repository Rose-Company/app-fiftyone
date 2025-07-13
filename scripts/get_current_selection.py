import fiftyone as fo
import os
import json
import argparse

def connect_database():
    """Kết nối tới MongoDB"""
    database_uri = os.getenv("FIFTYONE_DATABASE_URI")
    if not database_uri:
        print("Error: FIFTYONE_DATABASE_URI environment variable is required!")
        exit(1)
    fo.config.database_uri = database_uri

def get_current_selection(dataset_name, video_id=None):
    """
    Lấy selection hiện tại từ FiftyOne session đang chạy
    """
    connect_database()
    
    if not fo.dataset_exists(dataset_name):
        print(f"Dataset '{dataset_name}' không tồn tại")
        return None
    
    dataset = fo.load_dataset(dataset_name)
    
    # Lọc theo video_id nếu có
    if video_id:
        # Lọc dataset theo video_id
        filtered_view = dataset.match(fo.ViewField("video_id") == video_id)
        if len(filtered_view) == 0:
            # Thử lọc theo filepath nếu không có video_id field
            filtered_view = dataset.match(fo.ViewField("filepath").contains(f"/frames/{video_id}/"))
        
        if len(filtered_view) == 0:
            print(f"Không tìm thấy frames nào cho video_id: {video_id}")
            return None
        
        print(f"Tìm thấy {len(filtered_view)} frames cho video {video_id}")
        patches_view = filtered_view.to_patches("character_detections")
    else:
        # Tạo patches view cho toàn bộ dataset
        patches_view = dataset.to_patches("character_detections")
    
    # Lấy session hiện tại
    try:
        session = fo.launch_app(patches_view, remote=True, auto=False)
        
        print("Để lấy selection hiện tại:")
        print("1. Chọn patches trong FiftyOne UI")
        print("2. Nhấn Enter để lấy danh sách...")
        input()
        
        # Lấy selected samples
        selected_samples = session.selected
        
        if selected_samples:
            print(f"\nTìm thấy {len(selected_samples)} patches được chọn:")
            
            selected_info = []
            
            for i, sample_id in enumerate(selected_samples):
                try:
                    sample = patches_view[sample_id]
                    
                    # Lấy thông tin cơ bản
                    info = {
                        "index": i + 1,
                        "patch_id": str(sample.id),
                        "sample_id": str(sample.sample_id),
                        "character_id": None,
                        "detection_id": None
                    }
                    
                    # Lấy detection info
                    if sample.has_field("character_detections"):
                        detection = sample.character_detections
                        if detection:
                            info["detection_id"] = str(detection.id)
                            if detection.has_field("character_id"):
                                info["character_id"] = detection.character_id
                    
                    selected_info.append(info)
                    
                    # In thông tin ngắn gọn
                    print(f"  {i+1}. Detection ID: {info['detection_id']} | Character: {info['character_id']}")
                    
                except Exception as e:
                    print(f"  {i+1}. Lỗi: {e}")
            
            return selected_info
        else:
            print("Không có patches nào được chọn")
            return []
            
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

def parse_label_to_character_id(label):
    """
    Parse label thành character_id
    Ví dụ: "test11_character_3" -> character_id = "3", label = "test11_character_3"
    Ví dụ: "test1_unknown" -> character_id = "-1", label = "test1_unknown"
    Ví dụ: "unknown" -> character_id = "-1", label = "unknown"
    """
    # Xử lý trường hợp unknown (bao gồm cả _unknown)
    if label.lower() == "unknown" or label.lower().endswith("_unknown"):
        return "-1", label
    
    if "_character_" in label:
        parts = label.split("_character_")
        if len(parts) == 2:
            try:
                character_id = parts[1]
                return character_id, label
            except:
                pass
    
    # Nếu không parse được, sử dụng toàn bộ label làm character_id
    return label, label

def change_selected_to_character(dataset_name, new_label, video_id=None):
    """
    Lấy selection hiện tại và thay đổi character_id
    """
    character_id, label = parse_label_to_character_id(new_label)
    
    print(f"Sẽ thay đổi patches được chọn:")
    print(f"  Label: '{label}'")
    print(f"  Character ID: '{character_id}'")
    if video_id:
        print(f"  Video ID: '{video_id}'")
    
    selected_info = get_current_selection(dataset_name, video_id)
    
    if not selected_info:
        return False
    
    # Xác nhận
    print(f"\nBạn có muốn thay đổi {len(selected_info)} patches thành '{label}' (character_id: {character_id})? (y/n): ", end="")
    confirm = input().strip().lower()
    
    if confirm != 'y':
        print("Đã hủy")
        return False
    
    # Thực hiện thay đổi
    connect_database()
    dataset = fo.load_dataset(dataset_name)
    
    updated_count = 0
    
    for info in selected_info:
        try:
            sample = dataset[info["sample_id"]]
            
            if sample.has_field("character_detections"):
                for detection in sample.character_detections.detections:
                    if str(detection.id) == info["detection_id"]:
                        old_id = detection.character_id if detection.has_field("character_id") else "None"
                        old_label = detection.label if detection.has_field("label") else "None"
                        
                        # Cập nhật cả character_id và label
                        detection.character_id = character_id
                        if detection.has_field("label"):
                            detection.label = label
                        
                        sample.save()
                        
                        print(f"  {info['index']}. Đã thay đổi từ '{old_label}' (ID: {old_id}) thành '{label}' (ID: {character_id})")
                        updated_count += 1
                        break
        except Exception as e:
            print(f"  {info['index']}. Lỗi: {e}")
    
    print(f"\nĐã cập nhật {updated_count}/{len(selected_info)} patches")
    return updated_count

def export_detection_ids(dataset_name, output_file, video_id=None):
    """
    Xuất danh sách detection IDs của patches được chọn
    """
    selected_info = get_current_selection(dataset_name, video_id)
    
    if not selected_info:
        return False
    
    # Tạo danh sách detection IDs
    detection_ids = [info["detection_id"] for info in selected_info if info["detection_id"]]
    
    # Xuất ra file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_patches": len(selected_info),
            "detection_ids": detection_ids,
            "selected_info": selected_info
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nĐã xuất {len(detection_ids)} detection IDs ra file: {output_file}")
    
    # In ra command để sử dụng
    print("\nBạn có thể sử dụng detection IDs này với manual_face_editor.py:")
    for i, det_id in enumerate(detection_ids):
        print(f"docker-compose exec fiftyone python /app/scripts/manual_face_editor.py change-detection --detection-id \"{det_id}\" --new-id \"new_character_name\"")
        if i >= 2:  # Chỉ hiển thị 3 ví dụ đầu
            print(f"... và {len(detection_ids) - 3} detection IDs khác")
            break
    
    return True

def merge_characters(dataset_name, from_label, to_label, video_id=None):
    """
    Merge tất cả patches từ một character sang character khác trong video
    Ví dụ: merge "test11_character_7" to "test11_character_1"
    """
    connect_database()
    
    if not fo.dataset_exists(dataset_name):
        print(f"Dataset '{dataset_name}' không tồn tại")
        return False
    
    dataset = fo.load_dataset(dataset_name)
    
    # Parse labels
    from_character_id, from_label_parsed = parse_label_to_character_id(from_label)
    to_character_id, to_label_parsed = parse_label_to_character_id(to_label)
    
    print(f"Merge characters trong video '{video_id}':")
    print(f"  Từ: '{from_label_parsed}' (character_id: {from_character_id})")
    print(f"  Sang: '{to_label_parsed}' (character_id: {to_character_id})")
    
    # Lọc theo video_id
    if video_id:
        filtered_view = dataset.match(fo.ViewField("video_id") == video_id)
        if len(filtered_view) == 0:
            # Thử lọc theo filepath nếu không có video_id field
            filtered_view = dataset.match(fo.ViewField("filepath").contains(f"/frames/{video_id}/"))
        
        if len(filtered_view) == 0:
            print(f"Không tìm thấy frames nào cho video_id: {video_id}")
            return False
    else:
        filtered_view = dataset
    
    # Tìm tất cả patches có label và character_id khớp với from_label
    patches_to_merge = []
    
    for sample in filtered_view:
        if sample.has_field("character_detections"):
            for detection in sample.character_detections.detections:
                # Kiểm tra label và character_id
                detection_label = detection.label if detection.has_field("label") else ""
                detection_character_id = str(detection.character_id) if detection.has_field("character_id") else ""
                
                if (detection_label == from_label_parsed and 
                    detection_character_id == from_character_id):
                    patches_to_merge.append({
                        "sample_id": str(sample.id),
                        "detection_id": str(detection.id),
                        "current_label": detection_label,
                        "current_character_id": detection_character_id
                    })
    
    if not patches_to_merge:
        print(f"Không tìm thấy patches nào có label '{from_label_parsed}' và character_id '{from_character_id}' trong video '{video_id}'")
        return False
    
    print(f"\nTìm thấy {len(patches_to_merge)} patches cần merge")
    
    # Xác nhận
    print(f"\nBạn có muốn merge {len(patches_to_merge)} patches từ '{from_label_parsed}' sang '{to_label_parsed}'? (y/n): ", end="")
    confirm = input().strip().lower()
    
    if confirm != 'y':
        print("Đã hủy")
        return False
    
    # Thực hiện merge
    updated_count = 0
    
    for i, patch_info in enumerate(patches_to_merge):
        try:
            sample = dataset[patch_info["sample_id"]]
            
            if sample.has_field("character_detections"):
                for detection in sample.character_detections.detections:
                    if str(detection.id) == patch_info["detection_id"]:
                        # Cập nhật label và character_id
                        detection.label = to_label_parsed
                        detection.character_id = to_character_id
                        
                        sample.save()
                        
                        print(f"  {i+1}. Đã merge detection {patch_info['detection_id']} từ '{from_label_parsed}' sang '{to_label_parsed}'")
                        updated_count += 1
                        break
        except Exception as e:
            print(f"  {i+1}. Lỗi khi merge detection {patch_info['detection_id']}: {e}")
    
    print(f"\nĐã merge {updated_count}/{len(patches_to_merge)} patches")
    return updated_count > 0

def main():
    parser = argparse.ArgumentParser(description="Lấy selection hiện tại từ FiftyOne UI")
    
    # Cú pháp mới: video_id action command [args]
    parser.add_argument("video_id", help="Video ID cần xử lý")
    parser.add_argument("action", choices=["get", "update", "export", "merge"], help="Hành động thực hiện")
    parser.add_argument("command", nargs="?", help="Lệnh cụ thể")
    parser.add_argument("args", nargs="*", help="Tham số bổ sung")
    parser.add_argument("--dataset", default="video_dataset_faces_deepface_arcface_retinaface_final", help="Tên dataset")
    parser.add_argument("--output", help="File đầu ra (cho export)")
    
    args = parser.parse_args()
    
    if args.action == "get":
        get_current_selection(args.dataset, args.video_id)
    
    elif args.action == "update":
        if args.command == "label" and len(args.args) >= 1:
            new_label = args.args[0]
            change_selected_to_character(args.dataset, new_label, args.video_id)
        else:
            print("Cú pháp: video_id update label \"new_label\"")
            print("Ví dụ: test11 update label \"test11_character_3\"")
    
    elif args.action == "export":
        if args.output:
            export_detection_ids(args.dataset, args.output, args.video_id)
        else:
            # Tạo tên file tự động
            output_file = f"{args.video_id}_selected_patches.json"
            export_detection_ids(args.dataset, output_file, args.video_id)
    
    elif args.action == "merge":
        # Cú pháp: video_id merge "from_label" to "to_label"
        # args.command = "test11_character_7", args.args = ["to", "test11_character_1"]
        if args.command and len(args.args) >= 2 and args.args[0] == "to":
            from_label = args.command  # "test11_character_7"
            to_label = args.args[1]    # "test11_character_1"
            merge_characters(args.dataset, from_label, to_label, args.video_id)
        else:
            print("Cú pháp: video_id merge \"from_label\" to \"to_label\"")
            print("Ví dụ: test11 merge \"test11_character_7\" to \"test11_character_1\"")
    
    else:
        print("Cú pháp:")
        print("  python get_current_selection.py <video_id> get")
        print("  python get_current_selection.py <video_id> update label \"<new_label>\"")
        print("  python get_current_selection.py <video_id> export")
        print("  python get_current_selection.py <video_id> merge \"<from_label>\" to \"<to_label>\"")
        print("")
        print("Ví dụ:")
        print("  python get_current_selection.py test11 get")
        print("  python get_current_selection.py test11 update label \"test11_character_3\"")
        print("  python get_current_selection.py test11 export")
        print("  python get_current_selection.py test11 merge \"test11_character_7\" to \"test11_character_1\"")

if __name__ == "__main__":
    main() 