import cv2
import numpy as np
import dlib
import os
import gc
import fiftyone as fo
from fiftyone import ViewField as F

class DlibFaceRecognizer:
    def __init__(self, use_cnn=True):
        """
        Khởi tạo face recognizer sử dụng dlib trực tiếp
        Args:
            use_cnn: Sử dụng CNN (chính xác hơn) hoặc HOG (nhanh hơn)
        """
        # Tải các model của dlib
        if use_cnn:
            self.face_detector = dlib.cnn_face_detection_model_v1('/app/models/mmod_human_face_detector.dat')
        else:
            self.face_detector = dlib.get_frontal_face_detector()
        
        # Model tìm landmarks khuôn mặt
        self.shape_predictor = dlib.shape_predictor('/app/models/shape_predictor_68_face_landmarks.dat')
        
        # Model trích xuất face embedding
        self.face_recognizer = dlib.face_recognition_model_v1('/app/models/dlib_face_recognition_resnet_model_v1.dat')
        
        self.use_cnn = use_cnn
    
    def detect_faces(self, image):
        """
        Phát hiện khuôn mặt trong ảnh
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        if self.use_cnn:
            # CNN detector hoạt động tốt nhất với ảnh màu (BGR hoặc RGB).
            # Dlib Python API thường xử lý tốt ảnh BGR từ OpenCV.
            # Không cần chuyển sang grayscale cho CNN.
            # Model mong đợi một numpy array của ảnh.
            dets = self.face_detector(gray_image)  # Truyền ảnh màu gốc (BGR)
            # Chuyển mmod_rectangles sang định dạng dlib.rectangle thông thường
            # Đồng thời có thể lấy confidence nếu cần: faces = [(d.rect, d.confidence) for d in dets]
            faces = [d.rect for d in dets]
        else:
            # HOG detector hoạt động tốt hơn với ảnh grayscale
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                # Ảnh đã là grayscale
                gray_image = image
            faces = self.face_detector(gray_image)
            
        return faces
    
    def get_embedding(self, image, face):
        """
        Trích xuất embedding từ khuôn mặt đã phát hiện
        """
        # Tìm 68 điểm landmarks trên khuôn mặt
        shape = self.shape_predictor(image, face)
        
        # Trích xuất embedding 128 chiều
        face_descriptor = self.face_recognizer.compute_face_descriptor(image, shape)
        
        # Chuyển sang numpy array
        return np.array(face_descriptor)

def download_models():
    """
    Tải các model dlib nếu chưa có
    """
    models_dir = '/app/models'
    os.makedirs(models_dir, exist_ok=True)
    
    models = {
        'mmod_human_face_detector.dat': 'https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2',
        'shape_predictor_68_face_landmarks.dat': 'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2',
        'dlib_face_recognition_resnet_model_v1.dat': 'https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2'
    }
    
    for model_name, model_url in models.items():
        model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"Đang tải model {model_name}...")
            compressed_path = model_path + '.bz2'
            
            # Tải model
            import urllib.request
            urllib.request.urlretrieve(model_url, compressed_path)
            
            # Giải nén
            import bz2
            with bz2.open(compressed_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Xóa file nén
            os.remove(compressed_path)
            print(f"Đã tải xong model {model_name}")
        else:
            print(f"Model {model_name} đã tồn tại.")

def extract_face_from_rect(image, rect):
    """
    Trích xuất vùng khuôn mặt từ dlib rectangle
    """
    x = rect.left()
    y = rect.top()
    w = rect.width()
    h = rect.height()
    
    # Đảm bảo tọa độ nằm trong ảnh
    height, width = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    right = min(width, x + w)
    bottom = min(height, y + h)
    
    # Kiểm tra nếu rectangle không hợp lệ
    if y >= bottom or x >= right:
        print(f"Rectangle không hợp lệ: ({x},{y},{w},{h})")
        return None
    
    # Trích xuất vùng khuôn mặt
    face_image = image[y:bottom, x:right]
    return face_image

def rect_to_relative_bbox(rect, image_shape):
    """
    Chuyển đổi dlib rectangle sang bounding box tương đối
    """
    height, width = image_shape[:2]
    x = rect.left() / width
    y = rect.top() / height
    w = rect.width() / width
    h = rect.height() / height
    return [x, y, w, h]

def process_frames_with_dlib():
    """
    Xử lý frames có người và trích xuất đặc trưng khuôn mặt sử dụng dlib
    """
    print("=== Bắt đầu xử lý khuôn mặt với Dlib ===")
    
    # Tải các model cần thiết
    try:
        download_models()
    except Exception as e:
        print(f"Lỗi khi tải models: {e}")
        print("Đang sử dụng detector dlib mặc định (không CNN)...")
    
    # Kết nối MongoDB
    database_uri = os.getenv("FIFTYONE_DATABASE_URI")
    if not database_uri:
        print("Error: FIFTYONE_DATABASE_URI environment variable is required!")
        exit(1)
    fo.config.database_uri = database_uri
    fo.config.database_name = "fiftyone"
    
    # Load dataset frames
    frames_dataset_name = "my_images_17/6"
    if not fo.dataset_exists(frames_dataset_name):
        print(f"Không tìm thấy dataset '{frames_dataset_name}'")
        return
        
    frames_dataset = fo.load_dataset(frames_dataset_name)
    print(f"Đã load dataset '{frames_dataset_name}' với {len(frames_dataset)} frames")
    
    # Tạo dataset mới cho khuôn mặt
    face_dataset_name = "video_dataset_faces_dlib_test_cnn_17/6"
    if fo.dataset_exists(face_dataset_name):
        fo.delete_dataset(face_dataset_name)
    face_dataset = fo.Dataset(face_dataset_name)
    print(f"Đã tạo dataset mới '{face_dataset_name}'")
    
    # Xử lý theo batch để tiết kiệm bộ nhớ
    batch_size = 10
    total_frames = len(frames_dataset)
    batch_count = (total_frames + batch_size - 1) // batch_size
    total_faces = 0
    
    # Tạo face recognizer
    try:
        face_recognizer = DlibFaceRecognizer(use_cnn=True)
        using_cnn = True
        print("Sử dụng Dlib CNN face detector")
    except Exception as e:
        print(f"Không thể sử dụng CNN detector: {e}")
        face_recognizer = DlibFaceRecognizer(use_cnn=False)
        using_cnn = False
        print("Sử dụng Dlib HOG face detector")
    
    # Bắt đầu thời gian
    import time
    start_time = time.time()
    
    for i in range(batch_count):
        batch_start = time.time()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_frames)
        
        print(f"Xử lý batch {i+1}/{batch_count} (frames {start_idx+1}-{end_idx}/{total_frames})...")
        
        # Lấy batch samples
        batch_ids = frames_dataset.values("id")[start_idx:end_idx]
        batch_view = frames_dataset.select(batch_ids)
        
        for sample in batch_view:
            try:
                # Đọc ảnh từ disk
                image = cv2.imread(sample.filepath)
                
                if image is None:
                    print(f"Không thể đọc ảnh từ {sample.filepath}")
                    continue
                
                # Phát hiện khuôn mặt
                faces = face_recognizer.detect_faces(image)
                
                face_embeddings = []
                for face in faces:
                    # Trích xuất embedding
                    embedding = face_recognizer.get_embedding(image, face)
                    
                    # Tạo bounding box
                    bbox = rect_to_relative_bbox(face, image.shape)
                    
                    # Thêm vào danh sách
                    face_embeddings.append({
                        "embedding": embedding.tolist(),
                        "bounding_box": bbox,
                        "confidence": 1.0  # dlib không trả về confidence
                    })
                    total_faces += 1
                
                # Lưu thông tin khuôn mặt vào sample
                if face_embeddings:
                    sample["face_embeddings"] = face_embeddings
                    sample.save()
                    
                    # Thêm vào dataset khuôn mặt
                    face_dataset.add_sample(sample)
                
                # Giải phóng bộ nhớ
                del image
                gc.collect()
                
            except Exception as e:
                print(f"Lỗi khi xử lý frame {sample.filepath}: {str(e)}")
        
        # Thống kê thời gian cho batch này
        batch_time = time.time() - batch_start
        frames_per_sec = (end_idx - start_idx) / batch_time
        print(f"Batch {i+1} hoàn thành trong {batch_time:.2f}s ({frames_per_sec:.2f} frames/sec)")
        
        # Giải phóng bộ nhớ sau mỗi batch
        del batch_view
        gc.collect()
    
    # Tổng thời gian xử lý
    total_time = time.time() - start_time
    avg_time_per_frame = total_time / total_frames if total_frames > 0 else 0
    
    print(f"\n=== Kết quả xử lý khuôn mặt (Dlib {'CNN' if using_cnn else 'HOG'}) ===")
    print(f"- Tổng số frames xử lý: {total_frames}")
    print(f"- Tổng số khuôn mặt phát hiện được: {total_faces}")
    print(f"- Số frames có khuôn mặt: {len(face_dataset)}")
    print(f"- Tổng thời gian xử lý: {total_time:.2f}s")
    print(f"- Thời gian trung bình/frame: {avg_time_per_frame*1000:.2f}ms")
    print(f"- Tốc độ xử lý: {total_frames/total_time:.2f} frames/sec")
    
    # Tránh division by zero
    if total_frames > 0:
        face_ratio = len(face_dataset) / total_frames * 100
        print(f"- Tỷ lệ frames có khuôn mặt: {face_ratio:.2f}%")
    
    print(f"\nĐã lưu thông tin khuôn mặt vào dataset '{face_dataset_name}'")
    print("Truy cập http://localhost:" + os.getenv("FIFTYONE_PORT") + " để xem kết quả")

if __name__ == "__main__":
    process_frames_with_dlib()