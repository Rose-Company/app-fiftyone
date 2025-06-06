#!/usr/bin/env python3
# face_recognition_from_dataset.py

import os
import cv2
import numpy as np
import fiftyone as fo
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition
import argparse
from datetime import datetime

def load_embeddings_from_dataset(dataset_name="video_dataset_faces_dlib_test_2video"):
    """
    Tải các embedding đã có từ FiftyOne dataset
    """
    print(f"Đang kết nối đến MongoDB...")
    fo.config.database_uri = "mongodb://mongo:27017"
    
    if not fo.dataset_exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' không tồn tại")
    
    dataset = fo.load_dataset(dataset_name)
    print(f"Đã tải dataset '{dataset_name}' với {len(dataset)} samples")
    
    # Tạo từ điển lưu trữ tất cả các embedding theo nhân vật
    character_embeddings = {}
    character_sample_ids = {}
    
    # Đếm số mẫu có face_embeddings
    count = 0
    
    for sample in dataset:
        if sample.has_field("face_embeddings") and sample.face_embeddings:
            count += 1
            for face in sample.face_embeddings:
                if "embedding" in face and "character_id" in face:
                    character_id = face["character_id"]
                    embedding = np.array(face["embedding"])
                    
                    # Thêm embedding vào danh sách của nhân vật
                    if character_id not in character_embeddings:
                        character_embeddings[character_id] = []
                        character_sample_ids[character_id] = []
                    
                    character_embeddings[character_id].append(embedding)
                    character_sample_ids[character_id].append(sample.id)
    
    print(f"Đã tìm thấy {len(character_embeddings)} nhân vật từ {count} samples có face_embeddings")
    
    # Tính vector trung bình cho mỗi nhân vật
    character_centroids = {}
    for character_id, embeddings in character_embeddings.items():
        character_centroids[character_id] = np.mean(embeddings, axis=0)
        print(f"  Nhân vật {character_id}: {len(embeddings)} embeddings")
    
    return character_centroids, character_embeddings, character_sample_ids, dataset

def extract_face_embedding(image_path, model="hog"):
    """
    Trích xuất embedding từ khuôn mặt trong ảnh
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ: {image_path}")
    
    # Chuyển sang RGB (face_recognition cần định dạng RGB)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Tìm khuôn mặt trong ảnh
    print(f"Đang tìm khuôn mặt với model '{model}'...")
    face_locations = face_recognition.face_locations(rgb, model=model)
    
    if not face_locations:
        print("Không tìm thấy khuôn mặt nào trong ảnh")
        return [], [], image
    
    print(f"Đã tìm thấy {len(face_locations)} khuôn mặt")
    
    # Trích xuất embedding
    face_embeddings = face_recognition.face_encodings(rgb, face_locations)
    
    return face_embeddings, face_locations, image

def recognize_face(face_embedding, character_centroids, character_embeddings, threshold=0.6):
    """
    Nhận diện khuôn mặt bằng cách so sánh embedding với các nhân vật đã biết
    """
    best_match = None
    best_similarity = -1
    similarities = {}
    
    # So sánh với tâm của mỗi nhân vật
    for character_id, centroid in character_centroids.items():
        similarity = cosine_similarity([face_embedding], [centroid])[0][0]
        similarities[character_id] = similarity
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = character_id
    
    # Sắp xếp các nhân vật theo độ tương đồng giảm dần
    sorted_characters = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Nếu độ tương đồng vượt ngưỡng thì trả về kết quả
    if best_similarity >= threshold:
        return best_match, best_similarity, sorted_characters
    
    return None, best_similarity, sorted_characters

def visualize_results(query_image, face_locations, recognized_characters, dataset, character_sample_ids, output_dir="results", filename_prefix=""):
    """
    Hiển thị kết quả nhận diện
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo bản sao của ảnh để vẽ lên
    result_image = query_image.copy()
    
    # Vẽ các khuôn mặt được phát hiện và kết quả nhận diện
    for i, ((top, right, bottom, left), (character_id, similarity)) in enumerate(zip(face_locations, recognized_characters)):
        try:
            # Kiểm tra và đảm bảo tọa độ là số nguyên hợp lệ
            top, right, bottom, left = int(top), int(right), int(bottom), int(left)
            
            # Kiểm tra tọa độ hợp lệ (đủ 4 giá trị)
            if not all([isinstance(x, int) for x in [top, right, bottom, left]]):
                print(f"  Cảnh báo: Tọa độ khuôn mặt #{i+1} không hợp lệ: {top}, {right}, {bottom}, {left}")
                continue
                
            # Vẽ hình chữ nhật bao quanh khuôn mặt
            cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Hiển thị kết quả nhận diện
            label = f"Unknown ({similarity:.2f})" if character_id is None else f"Character {character_id} ({similarity:.2f})"
            y = top - 10 if top - 10 > 10 else top + 10
            cv2.putText(result_image, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Lấy mẫu đại diện của nhân vật (nếu có)
            if character_id is not None and character_id in character_sample_ids and character_sample_ids[character_id]:
                sample_id = character_sample_ids[character_id][0]  # Lấy mẫu đầu tiên
                sample = dataset[sample_id]
                if sample.has_field("filepath"):
                    ref_image_path = sample.filepath
                    try:
                        ref_image = cv2.imread(ref_image_path)
                        if ref_image is not None:
                            # Hiển thị ảnh đại diện
                            small_ref = cv2.resize(ref_image, (100, 100))
                            h, w = result_image.shape[:2]
                            
                            # Kiểm tra xem có đủ không gian để hiển thị ảnh tham chiếu không
                            if top + 100 < h and right + 110 < w:
                                result_image[top:top+100, right+10:right+110] = small_ref
                    except Exception as e:
                        print(f"  Không thể hiển thị ảnh tham chiếu: {str(e)}")
        except Exception as e:
            print(f"  Lỗi khi vẽ khuôn mặt #{i+1}: {str(e)}")
    
    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{filename_prefix}recognized_{timestamp}.jpg")
    cv2.imwrite(output_path, result_image)
    print(f"Đã lưu kết quả tại: {output_path}")
    
    return result_image, output_path

def process_dataset_images(dataset_name="video_dataset_faces_dlib_test_2video", model="hog", threshold=0.6, max_samples=None, save_visualization=True):
    """
    Xử lý và nhận diện khuôn mặt trong tất cả các ảnh từ dataset
    """
    print(f"=== BẮT ĐẦU NHẬN DIỆN KHUÔN MẶT TRONG DATASET ===")
    print(f"Dataset: {dataset_name}")
    print(f"Ngưỡng tương đồng: {threshold}")
    print(f"Số lượng mẫu tối đa: {'Không giới hạn' if max_samples is None else max_samples}")
    print()
    
    # Tải embedding từ dataset
    character_centroids, character_embeddings, character_sample_ids, dataset = load_embeddings_from_dataset(dataset_name)
    
    # Tạo view mới từ dataset - chỉ lấy các sample có face_embeddings
    samples_with_embeddings = dataset.match(F("face_embeddings").exists())
    print(f"Có {len(samples_with_embeddings)} mẫu có face_embeddings")
    
    # Giới hạn số lượng mẫu nếu cần
    if max_samples is not None:
        samples_to_process = samples_with_embeddings.limit(max_samples)
        print(f"Giới hạn xử lý {max_samples} mẫu")
    else:
        samples_to_process = samples_with_embeddings
    
    results = []
    
    # Đảm bảo dataset có trường lưu kết quả nhận diện
    if not dataset.has_sample_field("face_recognition_results"):
        dataset.add_sample_field(
            "face_recognition_results",
            fo.ListField,
            description="Kết quả nhận diện khuôn mặt"
        )
    
    # Xử lý từng mẫu
    for i, sample in enumerate(samples_to_process):
        print(f"\n--- Xử lý mẫu {i+1}/{len(samples_to_process)}: {sample.id} ---")
        
        if not sample.has_field("filepath") or not sample.filepath:
            print("  Bỏ qua: Không có đường dẫn file")
            continue
        
        if not sample.has_field("face_embeddings") or not sample.face_embeddings:
            print("  Bỏ qua: Không có face_embeddings")
            continue
        
        image_path = sample.filepath
        print(f"  Đang xử lý ảnh: {image_path}")
        
        try:
            # Đọc ảnh để hiển thị kết quả
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Bỏ qua: Không thể đọc ảnh từ {image_path}")
                continue
                
            # Lấy embedding và vị trí khuôn mặt từ sample
            face_embeddings = []
            face_locations = []
            face_indices = []  # Lưu index của face trong sample.face_embeddings
            
            for idx, face_data in enumerate(sample.face_embeddings):
                if "embedding" in face_data:
                    face_embeddings.append(np.array(face_data["embedding"]))
                    face_indices.append(idx)
                    
                    # Nếu có thông tin về vị trí khuôn mặt
                    if "bbox" in face_data:
                        bbox = face_data["bbox"]
                        # Chuyển đổi từ định dạng [x, y, width, height] sang [top, right, bottom, left]
                        x, y, w, h = bbox
                        top = y
                        right = x + w
                        bottom = y + h
                        left = x
                        face_locations.append((top, right, bottom, left))
                    else:
                        # Nếu không có thông tin bbox, sử dụng giá trị mặc định (toàn bộ ảnh)
                        h, w = image.shape[:2]
                        face_locations.append((0, w, h, 0))
            
            if not face_embeddings:
                print("  Bỏ qua: Không tìm thấy embedding khuôn mặt")
                continue
            
            # Lưu kết quả nhận diện cho sample
            face_recognition_results = []
            
            # Nhận diện từng khuôn mặt
            recognized_characters = []
            for j, (embedding, face_idx) in enumerate(zip(face_embeddings, face_indices)):
                character_id, similarity, sorted_characters = recognize_face(
                    embedding, character_centroids, character_embeddings, threshold)
                
                recognized_characters.append((character_id, similarity))
                
                print(f"  Khuôn mặt #{j+1}:")
                if character_id is not None:
                    print(f"    Nhận diện là: Nhân vật {character_id} (Độ tương đồng: {similarity:.4f})")
                else:
                    print(f"    Không nhận diện được (Độ tương đồng cao nhất: {similarity:.4f})")
                
                print("    Top 3 nhân vật tương đồng nhất:")
                top_matches = []
                for k, (char_id, sim) in enumerate(sorted_characters[:3]):
                    print(f"      #{k+1}: Nhân vật {char_id} (Độ tương đồng: {sim:.4f})")
                    top_matches.append({"character_id": char_id, "similarity": float(sim)})
                
                # Tạo kết quả nhận diện cho khuôn mặt này
                recognition_result = {
                    "face_index": face_idx,
                    "recognized_character_id": character_id,
                    "similarity": float(similarity),
                    "top_matches": top_matches
                }
                face_recognition_results.append(recognition_result)
                
                # Cập nhật character_id vào face_embeddings nếu nhận diện thành công
                if character_id is not None and similarity >= threshold:
                    sample.face_embeddings[face_idx]["recognized_character_id"] = character_id
                    sample.face_embeddings[face_idx]["recognition_similarity"] = float(similarity)
            
            # Lưu kết quả nhận diện vào sample
            sample.face_recognition_results = face_recognition_results
            sample.save()
            
            # Hiển thị kết quả nếu cần
            if save_visualization:
                filename_prefix = f"sample_{i+1}_"
                result_image, output_path = visualize_results(
                    image, face_locations, recognized_characters, 
                    dataset, character_sample_ids, filename_prefix=filename_prefix)
            
            # Lưu thông tin kết quả
            results.append({
                "sample_id": sample.id,
                "filepath": image_path,
                "faces_found": len(face_locations),
                "recognized_characters": recognized_characters
            })
            
        except Exception as e:
            print(f"  Lỗi khi xử lý mẫu {sample.id}: {str(e)}")
    
    # Lưu dataset sau khi hoàn tất
    dataset.save()
    print(f"\n=== KẾT THÚC NHẬN DIỆN KHUÔN MẶT TRONG DATASET ===")
    print(f"Đã xử lý {len(results)}/{len(samples_to_process)} mẫu thành công")
    print(f"Kết quả nhận diện đã được lưu vào dataset '{dataset_name}'")
    
    return results

def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Nhận diện khuôn mặt trong dataset")
    parser.add_argument("--dataset", default="video_dataset_faces_dlib_test_2video", 
                      help="Tên dataset chứa embedding (mặc định: video_dataset_faces_dlib_test_2video)")
    parser.add_argument("--model", default="hog", choices=["hog", "cnn"],
                      help="Mô hình phát hiện khuôn mặt (mặc định: hog)")
    parser.add_argument("--threshold", type=float, default=0.6,
                      help="Ngưỡng độ tương đồng (mặc định: 0.6)")
    parser.add_argument("--max-samples", type=int, default=None,
                      help="Số lượng mẫu tối đa cần xử lý (mặc định: không giới hạn)")
    parser.add_argument("--image", type=str, default=None,
                      help="Đường dẫn đến ảnh riêng lẻ cần nhận diện (tùy chọn)")
    parser.add_argument("--no-visualization", action="store_true",
                      help="Không lưu ảnh kết quả nhận diện")
    args = parser.parse_args()
    
    if args.image:
        # Chế độ nhận diện ảnh riêng lẻ
        print(f"=== BẮT ĐẦU NHẬN DIỆN KHUÔN MẶT TRÊN ẢNH RIÊNG LẺ ===")
        print(f"Ảnh đầu vào: {args.image}")
        print(f"Dataset: {args.dataset}")
        print(f"Mô hình: {args.model}")
        print(f"Ngưỡng tương đồng: {args.threshold}")
        print()
        
        # Tải embedding từ dataset
        character_centroids, character_embeddings, character_sample_ids, dataset = load_embeddings_from_dataset(args.dataset)
        
        # Trích xuất embedding từ ảnh đầu vào
        face_embeddings, face_locations, query_image = extract_face_embedding(args.image, args.model)
        
        # Nhận diện từng khuôn mặt
        recognized_characters = []
        for i, embedding in enumerate(face_embeddings):
            character_id, similarity, sorted_characters = recognize_face(
                embedding, character_centroids, character_embeddings, args.threshold)
            
            recognized_characters.append((character_id, similarity))
            
            print(f"\nKhuôn mặt #{i+1}:")
            if character_id is not None:
                print(f"  Nhận diện là: Nhân vật {character_id} (Độ tương đồng: {similarity:.4f})")
            else:
                print(f"  Không nhận diện được (Độ tương đồng cao nhất: {similarity:.4f})")
            
            print("  Top 3 nhân vật tương đồng nhất:")
            for j, (char_id, sim) in enumerate(sorted_characters[:3]):
                print(f"    #{j+1}: Nhân vật {char_id} (Độ tương đồng: {sim:.4f})")
        
        # Hiển thị kết quả
        result_image, output_path = visualize_results(
            query_image, face_locations, recognized_characters, dataset, character_sample_ids)
        
        print(f"\n=== KẾT THÚC NHẬN DIỆN KHUÔN MẶT ===")
        print(f"Đã lưu kết quả tại: {output_path}")
    else:
        # Chế độ nhận diện ảnh trong dataset
        process_dataset_images(args.dataset, args.model, args.threshold, 
                             args.max_samples, not args.no_visualization)

if __name__ == "__main__":
    import fiftyone as fo
    from fiftyone import ViewField as F
    
    main()