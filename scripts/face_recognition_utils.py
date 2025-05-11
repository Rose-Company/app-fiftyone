import cv2
import numpy as np
import face_recognition
from PIL import Image
import fiftyone as fo
from fiftyone import ViewField as F

class FaceRecognizer:
    def __init__(self, model="cnn"):
        """
        Khởi tạo face recognizer
        Args:
            model: "hog" (nhanh hơn) hoặc "cnn" (chính xác hơn)
        """
        self.model = model
        self.face_encodings = []
        self.face_names = []
        
    def get_embedding(self, face_image):
        """
        Trích xuất embedding từ ảnh khuôn mặt
        """
        # Chuyển đổi sang RGB nếu cần
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Tìm vị trí khuôn mặt
        face_locations = face_recognition.face_locations(face_image, model=self.model)
        if not face_locations:
            return None
            
        # Trích xuất embedding
        face_encodings = face_recognition.face_encodings(face_image, face_locations)
        return face_encodings[0] if face_encodings else None

def extract_face_from_bbox(image, bbox):
    """
    Trích xuất vùng khuôn mặt từ bounding box
    """
    # Chuyển đổi bbox từ [x, y, width, height] (tỷ lệ tương đối)
    # sang [top, right, bottom, left] (tọa độ pixel)
    height, width = image.shape[:2]
    x, y, w, h = bbox
    top = int(y * height)
    left = int(x * width)
    bottom = int((y + h) * height)
    right = int((x + w) * width)
    
    # Đảm bảo tọa độ nằm trong ảnh
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    left = max(0, left)
    
    # Kiểm tra nếu bounding box không hợp lệ
    if top >= bottom or left >= right:
        print(f"Bounding box không hợp lệ: {bbox}")
        return None
    
    # Trích xuất vùng khuôn mặt
    face_image = image[top:bottom, left:right]
    return face_image

def process_frames_with_faces():
    """
    Xử lý frames có người và trích xuất đặc trưng khuôn mặt
    """
    # Kết nối MongoDB
    fo.config.database_uri = "mongodb://mongo:27017"
    
    # Load dataset có người
    person_dataset_name = "video_dataset_person_frames"
    if not fo.dataset_exists(person_dataset_name):
        print(f"Không tìm thấy dataset '{person_dataset_name}'")
        return
        
    person_dataset = fo.load_dataset(person_dataset_name)
    print(f"Đã load dataset '{person_dataset_name}' với {len(person_dataset)} frames")
    
    # Khởi tạo face recognizer
    face_recognizer = FaceRecognizer(model="cnn")  # Sử dụng HOG cho tốc độ nhanh hơn
    
    # Tạo dataset mới cho khuôn mặt
    face_dataset_name = "video_dataset_faces"
    if fo.dataset_exists(face_dataset_name):
        fo.delete_dataset(face_dataset_name)
    face_dataset = fo.Dataset(face_dataset_name)
    
    # Xử lý từng frame
    total_faces = 0
    for sample in person_dataset:
        # Lấy ảnh frame
        image = cv2.imread(sample.filepath)
        
        if image is None:
            print(f"Không thể đọc ảnh từ {sample.filepath}")
            continue
        
        # Chuyển từ BGR sang RGB nếu cần
        if image is not None and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Lấy các detection người
        detections = sample.detections.detections
        face_embeddings = []
        
        for det in detections:
            if det.label == "person":
                # Trích xuất vùng khuôn mặt
                face_crop = extract_face_from_bbox(image, det.bounding_box)
                
                if face_crop is None or face_crop.size == 0:
                    print(f"Không thể trích xuất khuôn mặt từ bounding box: {det.bounding_box}")
                    continue
                
                print(f"Bounding box: {det.bounding_box}")
                print(f"Kích thước ảnh: {image.shape}")
                
                # Trích xuất embedding
                embedding = face_recognizer.get_embedding(face_crop)
                
                if embedding is not None:
                    face_embeddings.append({
                        "embedding": embedding.tolist(),
                        "bbox": det.bounding_box,
                        "confidence": det.confidence
                    })
                    total_faces += 1
        
        # Lưu thông tin khuôn mặt vào frame
        if face_embeddings:
            sample["face_embeddings"] = face_embeddings
            sample.save()
            
            # Thêm vào dataset khuôn mặt
            face_dataset.add_sample(sample)
    
    print(f"\nKết quả xử lý khuôn mặt:")
    print(f"- Tổng số frames xử lý: {len(person_dataset)}")
    print(f"- Tổng số khuôn mặt phát hiện được: {total_faces}")
    print(f"- Số frames có khuôn mặt: {len(face_dataset)}")
    
    print(f"\nĐã lưu thông tin khuôn mặt vào dataset '{face_dataset_name}'")
    print("Truy cập http://localhost:5151 để xem kết quả")

if __name__ == "__main__":
    process_frames_with_faces()

