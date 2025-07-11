#!/usr/bin/env python3
# extract_custom_dlib_embedding.py

import fiftyone as fo
import numpy as np
import argparse
from tqdm import tqdm

def extract_embeddings(dataset_name):
    """
    Trích xuất trường custom_dlib_embedding từ face_embeddings và đưa ra ngoài cùng
    """
    print(f"Đang kết nối đến MongoDB...")
    fo.config.database_uri = "mongodb://mongo:27017"
    
    if not fo.dataset_exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' không tồn tại")
    
    dataset = fo.load_dataset(dataset_name)
    print(f"Đã tải dataset '{dataset_name}' với {len(dataset)} samples")
    
    # Lọc các sample có face_embeddings
    samples_with_embeddings = dataset.match(F("face_embeddings").exists())
    print(f"Có {len(samples_with_embeddings)} mẫu có face_embeddings")
    
    # Đảm bảo dataset có trường mới để lưu embeddings
    if not dataset.has_sample_field("face_embeddings_extracted"):
        dataset.add_sample_field(
            "face_embeddings_extracted",
            fo.EmbeddedDocumentField,
            description="Face embeddings extracted from face_embeddings"
        )
    
    count = 0
    # Xử lý từng sample
    for sample in tqdm(samples_with_embeddings):
        # Kiểm tra xem sample có face_embeddings không
        if not hasattr(sample, "face_embeddings") or not sample.face_embeddings:
            continue
            
        # Trường hợp face_embeddings là một object (không phải list)
        if isinstance(sample.face_embeddings, dict) or not isinstance(sample.face_embeddings, list):
            # Kiểm tra nếu có trường custom_dlib_embedding
            if hasattr(sample.face_embeddings, "custom_dlib_embedding"):
                # Lưu trực tiếp embedding vào trường mới
                sample.face_embeddings_extracted = sample.face_embeddings.custom_dlib_embedding
                sample.save()
                count += 1
        else:
            # Trường hợp face_embeddings là một list
            # Trong trường hợp này, chỉ lấy embedding từ phần tử đầu tiên có custom_dlib_embedding
            for face in sample.face_embeddings:
                # Kiểm tra nếu face là dict
                if isinstance(face, dict) and "custom_dlib_embedding" in face:
                    sample.face_embeddings_extracted = face["custom_dlib_embedding"]
                    sample.save()
                    count += 1
                    break
                # Kiểm tra nếu face là object Detection
                elif hasattr(face, "custom_dlib_embedding"):
                    sample.face_embeddings_extracted = face.custom_dlib_embedding
                    sample.save()
                    count += 1
                    break
    
    # Lưu dataset sau khi hoàn tất
    dataset.save()
    print(f"Đã xử lý {count}/{len(samples_with_embeddings)} mẫu thành công")
    print(f"Kết quả đã được lưu vào dataset '{dataset_name}' trong trường 'face_embeddings_extracted'")
    
    return dataset

def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Trích xuất custom_dlib_embedding từ face_embeddings")
    parser.add_argument("--dataset", default="2025.06.06.05.09.37.681359", 
                      help="Tên dataset cần xử lý (mặc định: 2025.06.06.05.09.37.681359)")
    args = parser.parse_args()
    
    print(f"=== BẮT ĐẦU TRÍCH XUẤT CUSTOM_DLIB_EMBEDDING ===")
    print(f"Dataset: {args.dataset}")
    
    # Thực hiện trích xuất
    dataset = extract_embeddings(args.dataset)
    
    print(f"\n=== KẾT THÚC TRÍCH XUẤT CUSTOM_DLIB_EMBEDDING ===")
    print(f"Dữ liệu đã được lưu vào trường 'face_embeddings_extracted'")
    
if __name__ == "__main__":
    import fiftyone as fo
    from fiftyone import ViewField as F
    
    main() 