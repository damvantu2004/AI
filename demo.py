import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import pandas as pd
from model_utils import mock_predict, class_names, get_prediction_color

def main():
    # Set page config
    st.set_page_config(
        page_title="Driver Behavior Detection Demo",
        page_icon="🚗",
        layout="wide"
    )

    # Sidebar
    with st.sidebar:
        st.header("🔧 Thông tin")
        st.info("""
        **Demo Mode**
        - Không sử dụng mô hình AI
        - Kết quả là giả lập
        - Dùng để test giao diện
        """)
        
        st.header("📝 Hướng dẫn")
        st.write("""
        1. Upload ảnh hoặc video
        2. Xem kết quả phân tích
        3. Với video: dùng slider để xem từng frame
        """)
        
        st.header("🎯 Classes")
        for class_name in class_names:
            st.markdown(
                f'<div style="color:{get_prediction_color(class_name)}">'
                f'• {class_name}</div>', 
                unsafe_allow_html=True
            )

    # Main content
    st.title("Driver Behavior Detection 🚗")
    st.write("##### Tải ảnh hoặc video lên để phân tích hành vi tài xế.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Chọn ảnh hoặc video...", 
        type=["jpg", "jpeg", "png", "mp4"]
    )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Xử lý ảnh
        if file_extension in ["jpg", "jpeg", "png"]:
            process_image(uploaded_file)

        # Xử lý video
        elif file_extension == "mp4":
            process_video(uploaded_file)

def process_image(uploaded_file):
    """Xử lý và hiển thị kết quả cho ảnh"""
    # Hiển thị ảnh
    col1, col2 = st.columns([2, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã tải lên', use_column_width=True)

    with col2:
        st.write("**Thông tin ảnh:**")
        st.write(f"- Kích thước: {image.size[0]} x {image.size[1]}")
        st.write(f"- Mode: {image.mode}")
        st.write(f"- Format: {image.format}")

    # Nút phân tích
    if st.button("🔍 Phân tích ảnh"):
        with st.spinner('Đang phân tích...'):
            predicted_class, predictions = mock_predict(image)
        
        # Hiển thị kết quả
        display_results(predicted_class, predictions[0])

def process_video(uploaded_file):
    """Xử lý và hiển thị kết quả cho video"""
    # Lưu video tạm thời
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    # Hiển thị video
    st.video(tfile.name)

    # Lấy thông tin video
    video = cv2.VideoCapture(tfile.name)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Hiển thị thông tin video
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Số frames", f"{total_frames}")
    with col2:
        st.metric("FPS", f"{fps}")
    with col3:
        st.metric("Thời lượng", f"{duration:.1f}s")
    with col4:
        st.metric("Độ phân giải", f"{width}x{height}")

    if st.button("🎥 Phân tích video"):
        analyze_video(video, total_frames)
        video.release()
        os.remove(tfile.name)

def analyze_video(video, total_frames):
    """Phân tích từng frame của video"""
    frames_to_analyze = min(10, total_frames)  # Giới hạn 10 frames để demo
    
    predictions = []
    frames = []
    
    with st.spinner(f'Đang phân tích {frames_to_analyze} frames...'):
        for i in range(frames_to_analyze):
            ret, frame = video.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            predicted_class, prediction = mock_predict(image)
            predictions.append(prediction[0])
            frames.append(frame_rgb)
    
    # Lưu kết quả vào session state
    st.session_state.predictions = predictions
    st.session_state.frames = frames
    st.session_state.total_analyzed = len(predictions)
    
    display_video_results()

def display_video_results():
    """Hiển thị kết quả phân tích video"""
    if 'predictions' not in st.session_state:
        return

    st.write("### 🎬 Kết quả phân tích video")
    
    # Slider để chọn frame
    frame_idx = st.slider(
        "Chọn frame", 
        0, 
        st.session_state.total_analyzed - 1, 
        0
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(
            st.session_state.frames[frame_idx], 
            caption=f'Frame {frame_idx + 1}', 
            use_column_width=True
        )
    
    with col2:
        display_results(
            class_names[np.argmax(st.session_state.predictions[frame_idx])],
            st.session_state.predictions[frame_idx]
        )

def display_results(predicted_class, probabilities):
    """Hiển thị kết quả dự đoán"""
    st.write("### 📊 Kết quả phân tích")
    
    # Hiển thị class được dự đoán
    st.markdown(
        f'<h4 style="color:{get_prediction_color(predicted_class)}">'
        f'🎯 Dự đoán: {predicted_class}</h4>', 
        unsafe_allow_html=True
    )
    
    # Hiển thị xác suất cho từng class
    st.write("**Chi tiết xác suất:**")
    for class_name, prob in zip(class_names, probabilities):
        color = get_prediction_color(class_name)
        st.markdown(
            f'<div style="color:{color}">{class_name}:</div>',
            unsafe_allow_html=True
        )
        st.progress(float(prob))
        st.write(f'{prob:.3f}')

    # Hiển thị dưới dạng DataFrame
    df = pd.DataFrame([probabilities], columns=class_names)
    st.write("**Bảng xác suất:**")
    st.dataframe(df.style.format("{:.3f}"))
    
    # Vẽ biểu đồ
    st.write("**Biểu đồ xác suất:**")
    st.bar_chart(df.T)

if __name__ == "__main__":
    main()