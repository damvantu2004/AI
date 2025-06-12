# Driver Behavior Detection 🚗🤖

Ứng dụng phát hiện hành vi người điều khiển xe ô tô sử dụng học sâu (Deep Learning) với giao diện web thân thiện.

## 📋 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cách chạy](#cách-chạy)
- [Cách sử dụng](#cách-sử-dụng)
- [Hiểu về code](#hiểu-về-code)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Mô hình](#mô-hình)
- [Đóng góp](#đóng-góp)

## 🎯 Giới thiệu

Dự án này sử dụng các mô hình CNN (Convolutional Neural Networks) để phân tích và phân loại hành vi của tài xế thông qua ảnh hoặc video. Mục tiêu là tăng cường an toàn giao thông bằng cách phát hiện các hành vi nguy hiểm như nhắn tin, nói chuyện điện thoại khi lái xe.

### Các lớp hành vi được phân loại:
- **Other**: Hành vi khác
- **Safe**: Lái xe an toàn
- **Talking**: Đang nói chuyện (điện thoại)
- **Texting**: Đang nhắn tin
- **Turning**: Đang rẽ/quay đầu

## ✨ Tính năng

- 🖼️ **Phân tích ảnh**: Upload và phân tích ảnh tài xế
- 🎥 **Phân tích video**: Xử lý video theo từng khung hình
- 🧠 **Đa mô hình**: So sánh 3 mô hình CNN khác nhau (AlexNet, VGG16, GoogLeNet)
- 📊 **Hiển thị xác suất**: Xem độ tin cậy của từng dự đoán
- 🌐 **Giao diện web**: Interface thân thiện bằng Streamlit
- 🇻🇳 **Tiếng Việt**: Giao diện hoàn toàn bằng tiếng Việt

## 💻 Yêu cầu hệ thống

- Python 3.7+
- RAM: Tối thiểu 4GB (khuyến nghị 8GB+)
- GPU: Không bắt buộc nhưng sẽ tăng tốc độ xử lý
- Dung lượng: ~2GB cho các mô hình

## 🚀 Cài đặt

1. **Clone repository:**
```bash
git clone <repository-url>
cd DRIVER-BEHAVIOR-main
```

2. **Tạo virtual environment (khuyến nghị):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

3. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

4. **Kiểm tra các mô hình:**
Đảm bảo các file mô hình trong thư mục `model/`:
- `alex_model.h5`
- `vgg16_model.h5` 
- `inception_model.h5`

## 🏃‍♂️ Cách chạy

1. **Khởi động ứng dụng:**
```bash
streamlit run app.py
```

2. **Mở trình duyệt:**
- Ứng dụng sẽ tự động mở tại `http://localhost:8501`
- Nếu không tự mở, copy link từ terminal

3. **Dừng ứng dụng:**
- Nhấn `Ctrl + C` trong terminal

## 📖 Cách sử dụng

### Phân tích ảnh:
1. Click "Browse files" để upload ảnh (JPG, PNG)
2. Chọn mô hình muốn sử dụng
3. Click "Dự đoán"
4. Xem kết quả và xác suất

### Phân tích video:
1. Upload file video (MP4)
2. Chọn mô hình
3. Click "Dự đoán" (có thể mất vài phút)
4. Sử dụng slider để xem từng khung hình
5. Xem dự đoán cho từng khung hình

## 🔍 Hiểu về code

### Cấu trúc chính của `app.py`:

#### 1. Import và Load mô hình
```python
import streamlit as st
import tensorflow as tf
# ... các import khác

# Load 3 mô hình đã được train
alex_model = load_model('model/alex_model.h5')
vgg16_model = load_model('model/vgg16_model.h5')
inception_model = load_model('model/inception_model.h5')
```

#### 2. Tiền xử lý ảnh
```python
def preprocess_image(image, target_size=(224, 224)):
    # Chuyển sang RGB nếu cần
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize về kích thước chuẩn
    img = image.resize(target_size)
    
    # Chuyển sang array và thêm batch dimension
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    # Normalize về [0,1]
    img = img / 255.0
    return img
```

#### 3. Hàm dự đoán
```python
def predict(model, image):
    # Tiền xử lý ảnh
    processed_image = preprocess_image(image)
    
    # Dự đoán với mô hình
    predictions = model.predict(processed_image)
    
    # Lấy class có xác suất cao nhất
    predicted_class = class_names[np.argmax(predictions)]
    
    return predicted_class, predictions
```

#### 4. Giao diện Streamlit
- `st.file_uploader()`: Upload file
- `st.selectbox()`: Chọn mô hình
- `st.button()`: Nút dự đoán
- `st.image()`: Hiển thị ảnh
- `st.video()`: Hiển thị video
- `st.slider()`: Điều khiển khung hình video

#### 5. Xử lý video
- Sử dụng OpenCV để đọc từng frame
- Chuyển đổi BGR sang RGB
- Dự đoán cho từng frame
- Lưu kết quả vào session state

### Workflow xử lý:
1. **Upload** → 2. **Detect file type** → 3. **Preprocess** → 4. **Model predict** → 5. **Display results**

## 📁 Cấu trúc dự án

## 🧠 Mô hình

### 1. AlexNet
- **Kiến trúc**: 8 layers (5 conv + 3 dense)
- **Kích thước input**: 224x224x3
- **Ưu điểm**: Nhanh, phù hợp real-time
- **Nhược điểm**: Độ chính xác thấp hơn

### 2. VGG16
- **Kiến trúc**: 16 layers với conv 3x3
- **Kích thước input**: 224x224x3
- **Ưu điểm**: Độ chính xác cao, ổn định
- **Nhược điểm**: Nặng, chậm hơn

### 3. GoogLeNet (Inception V1)
- **Kiến trúc**: Inception modules
- **Kích thước input**: 224x224x3
- **Ưu điểm**: Cân bằng tốc độ và độ chính xác
- **Nhược điểm**: Phức tạp hơn

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **ModuleNotFoundError**:
```bash
pip install -r requirements.txt
```

2. **Mô hình không load được**:
- Kiểm tra file .h5 trong thư mục model/
- Đảm bảo TensorFlow version tương thích

3. **Out of memory**:
- Giảm kích thước batch
- Sử dụng ảnh có resolution thấp hơn

4. **Video không xử lý được**:
- Kiểm tra codec video
- Thử convert sang MP4 standard

## 📈 Cải tiến có thể

- [ ] Thêm mô hình YOLO cho object detection
- [ ] Hỗ trợ webcam real-time
- [ ] API endpoint cho mobile app
- [ ] Thêm các metrics đánh giá mô hình
- [ ] Export kết quả dưới dạng CSV/JSON

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Liên hệ

- Email: your.email@example.com
- Project Link: [https://github.com/yourusername/driver-behavior](https://github.com/yourusername/driver-behavior)

---

**⚠️ Lưu ý**: Dự án này chỉ mang tính chất học tập và nghiên cứu. Không sử dụng cho mục đích thương mại hoặc trong môi trường production mà không có kiểm thử kỹ lưỡng.

Chọn VGG16 nếu:
Độ chính xác là ưu tiên số 1
Có đủ tài nguyên máy tính
Không quan tâm tốc độ xử lý
Dữ liệu test phức tạp
Chọn AlexNet nếu:
Tài nguyên máy rất hạn chế
Cần tốc độ cực nhanh
Chấp nhận độ chính xác thấp hơn
Prototype/demo nhanh
🚀 Kết luận:
GoogLeNet (Inception V1) là lựa chọn tối ưu nhất vì:
Sweet spot giữa performance và efficiency
Practical cho ứng dụng thực tế
Scalable dễ mở rộng và triển khai
Modern architecture với Inception modules