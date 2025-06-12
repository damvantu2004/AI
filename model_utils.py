import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Định nghĩa classes
class_names = ['Other', 'Safe', 'Talking', 'Texting', 'Turning']

def preprocess_image(image, target_size=(224, 224)):
    """Tiền xử lý ảnh"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def mock_predict(image):
    """Hàm dự đoán giả lập cho demo"""
    # Tạo xác suất ngẫu nhiên có weighted bias
    base_probs = np.random.dirichlet(np.ones(5), size=1)[0]
    
    # Thêm một chút bias cho class 'Safe' (index 1)
    base_probs[1] *= 1.2
    base_probs = base_probs / base_probs.sum()  # Normalize lại
    
    # Chọn class dựa trên xác suất
    predicted_class = class_names[np.argmax(base_probs)]
    
    return predicted_class, base_probs.reshape(1, -1)

def get_prediction_color(class_name):
    """Trả về màu cho từng loại dự đoán"""
    color_map = {
        'Safe': '#28a745',      # Xanh lá - An toàn
        'Texting': '#dc3545',   # Đỏ - Nguy hiểm
        'Talking': '#ffc107',   # Vàng - Cảnh báo
        'Turning': '#17a2b8',   # Xanh dương - Thông tin
        'Other': '#6c757d'      # Xám - Khác
    }
    return color_map.get(class_name, '#6c757d')
