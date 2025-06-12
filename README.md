# Driver Behavior Detection 🚗🤖

Ứng dụng phát hiện hành vi người điều khiển xe ô tô sử dụng học sâu (Deep Learning) với giao diện web thân thiện, được triển khai bằng Streamlit.

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
- 🧠 **Đa mô hình**: So sánh 3 mô hình CNN khác nhau (AlexNet, InceptionV3, GoogLeNet)
- 📊 **Hiển thị xác suất**: Xem độ tin cậy của từng dự đoán
- 🌐 **Giao diện web**: Interface thân thiện bằng Streamlit
- 🚀 **Tự động tải model**: Tự động tải các mô hình từ Google Drive trong lần chạy đầu tiên.
- 🇻🇳 **Tiếng Việt**: Giao diện hoàn toàn bằng tiếng Việt

## 💻 Yêu cầu hệ thống

- Python 3.7+
- RAM: Tối thiểu 4GB (khuyến nghị 8GB+)
- GPU: Không bắt buộc nhưng sẽ tăng tốc độ xử lý
- Dung lượng: ~250MB cho các mô hình và dependencies.

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
Thư viện `gdown` đã được thêm vào `requirements.txt` để tải model.

4. **Chuẩn bị mô hình:**
Không cần thao tác thủ công. Các tệp mô hình sẽ được **tự động tải xuống từ Google Drive** khi bạn khởi động ứng dụng lần đầu tiên.

## 🏃‍♂️ Cách chạy

1. **Khởi động ứng dụng:**
```bash
streamlit run app.py
```

2. **Mở trình duyệt:**
- Ứng dụng sẽ tự động mở tại `http://localhost:8501`
- **Lưu ý**: Trong lần chạy đầu tiên, ứng dụng sẽ mất một lúc để tải các mô hình (~250MB). Vui lòng chờ cho đến khi quá trình tải hoàn tất.

3. **Dừng ứng dụng:**
- Nhấn `Ctrl + C` trong terminal

## 📖 Cách sử dụng

### Phân tích ảnh:
1. Click "Browse files" để upload ảnh (JPG, PNG)
2. Chọn mô hình muốn sử dụng từ thanh bên trái
3. Xem kết quả và biểu đồ phân tích độ tin cậy

### Phân tích video:
1. Upload file video (MP4)
2. Chọn mô hình
3. Sử dụng slider để xem dự đoán cho từng khung hình

## 🔍 Hiểu về code

### Cấu trúc chính của `app.py`:

#### 1. Tải và cache mô hình
Ứng dụng sử dụng `st.cache_resource` để tải và lưu trữ các mô hình, đồng thời tự động tải chúng từ Google Drive nếu chưa có.

```python
import streamlit as st
import os
import gdown
from tensorflow.keras.models import load_model

@st.cache_resource
def load_models_safely():
    # ... (Khai báo thông tin model và ID Google Drive)
    
    for name, info in model_info.items():
        # Tải model nếu chưa tồn tại
        if not os.path.exists(info['path']):
            st.info(f"Downloading {name} model...")
            gdown.download(id=info['id'], output=info['path'], quiet=False)
        
        # Tải mô hình vào bộ nhớ
        try:
            models[name] = load_model(info['path'])
        except Exception as e:
            # ... (Xử lý lỗi)
    return models, model_info
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
def predict_with_confidence(model, image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    # ... (Xử lý và trả về kết quả)
```

## 🧠 Mô hình

### 1. AlexNet
- **Kiến trúc**: 8 layers (5 conv + 3 dense)
- **Ưu điểm**: Nhẹ, nhanh, phù hợp cho các tác vụ cần tốc độ cao hoặc trên các thiết bị cấu hình thấp.
- **Nhược điểm**: Độ chính xác thấp hơn so với các mô hình hiện đại.

### 2. InceptionV3
- **Kiến trúc**: Sử dụng các "Inception module" để xử lý song song các filter với kích thước khác nhau.
- **Ưu điểm**: Độ chính xác cao, hiệu quả về mặt tính toán hơn VGG.
- **Nhược điểm**: Nặng hơn AlexNet và GoogLeNet (phiên bản trong dự án này).

### 3. GoogLeNet (Inception V1)
- **Kiến trúc**: Phiên bản đầu tiên của kiến trúc Inception.
- **Ưu điểm**: Đạt được sự cân bằng rất tốt giữa tốc độ, kích thước model và độ chính xác.
- **Nhược điểm**: Phức tạp hơn AlexNet.

### Lựa chọn nào là tốt nhất?
**GoogLeNet (Inception V1)** thường là lựa chọn tối ưu nhất trong dự án này vì nó cung cấp một điểm cân bằng tuyệt vời giữa hiệu suất (độ chính xác) và hiệu quả (tốc độ, kích thước), rất phù hợp cho các ứng dụng thực tế.

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **ModuleNotFoundError**:
   Hãy chắc chắn bạn đã chạy:
   ```bash
   pip install -r requirements.txt
   ```

2. **Lỗi tải mô hình**:
   - Kiểm tra kết nối Internet.
   - Đảm bảo các ID file trên Google Drive trong `app.py` là chính xác và tệp được chia sẻ công khai.

3. **Out of memory (Hết bộ nhớ)**:
   - Khởi động lại ứng dụng.
   - Nếu deploy, hãy chọn một nền tảng cung cấp đủ RAM (tối thiểu 2GB).

## 📈 Cải tiến có thể

- [ ] Hỗ trợ webcam real-time
- [ ] API endpoint cho các ứng dụng khác
- [ ] Export kết quả phân tích ra file CSV/JSON

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

---

**⚠️ Lưu ý**: Dự án này chỉ mang tính chất học tập và nghiên cứu. Không sử dụng cho mục đích thương mại hoặc trong môi trường production mà không có kiểm thử kỹ lưỡng.