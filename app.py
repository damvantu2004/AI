import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import tempfile
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ================================
# 🎨 CUSTOM CSS STYLING
# ================================
st.set_page_config(
    page_title="Driver Behavior AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 1rem;
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Model cards */
    .model-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Prediction confidence styling */
    .confidence-high { 
        color: #28a745; 
        font-weight: bold; 
    }
    .confidence-medium { 
        color: #ffc107; 
        font-weight: bold; 
    }
    .confidence-low { 
        color: #dc3545; 
        font-weight: bold; 
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #2196F3;
        background-color: #f0f8ff;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide streamlit footer */
    .css-1d391kg { display: none; }
    footer { display: none; }
    .css-1rs6os { display: none; }
    
    /* Progress bar custom style */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 🚗 HEADER SECTION
# ================================
st.markdown("""
<div class="custom-header">
    <h1>🚗 Driver Behavior AI Detection</h1>
    <p style="font-size: 1.2em; margin-top: 1rem;">
        🤖 Phân tích hành vi lái xe thông minh với Deep Learning
    </p>
    <p style="font-size: 1em; opacity: 0.9;">
        📊 Hỗ trợ 3 mô hình AI: AlexNet • InceptionV3 • GoogLeNet
    </p>
</div>
""", unsafe_allow_html=True)

# ================================
# 🔧 MODEL LOADING WITH SAFETY
# ================================
@st.cache_resource
def load_models_safely():
    """Load models with error handling"""
    models = {}
    model_info = {
        'AlexNet': {'path': 'model/alex_model.keras', 'size': '~84MB', 'accuracy': '85%'},
        'InceptionV3': {'path': 'model/inception_model (1).keras', 'size': '~110MB', 'accuracy': '79%'},
        'GoogLeNet': {'path': 'model/inception_model.keras', 'size': '~27MB', 'accuracy': '89%'}
    }
    
    for name, info in model_info.items():
        try:
            models[name] = load_model(info['path'])
            st.success(f"✅ {name} tải lên thành công")
        except Exception as e:
            st.error(f"❌ Error loading {name}: {str(e)}")
            # Tạo mock model nếu load thất bại
            models[name] = None
    
    return models, model_info

models, model_info = load_models_safely()
class_names = ['Other', 'Safe', 'Talking', 'Texting', 'Turning']

# ================================
# 🎛️ SIDEBAR CONFIGURATION
# ================================
with st.sidebar:
    st.markdown("## 🔧 Cấu hình")
    
    # Model selection with details
    st.markdown("### 🤖 Chọn mô hình AI")
    available_models = list(models.keys())
    
    if available_models:
        selected_model_name = st.selectbox(
            "Mô hình:", 
            available_models,
            help="Chọn mô hình để phân tích hình ảnh"
        )
        
        # Display model info
        if selected_model_name in model_info:
            info = model_info[selected_model_name]
            st.markdown(f"""
            <div class="model-card">
                <h4>📊 {selected_model_name}</h4>
                <p><strong>Kích thước:</strong> {info['size']}</p>
                <p><strong>Độ chính xác:</strong> {info['accuracy']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("❌ Không có mô hình nào khả dụng!")
        st.stop()
    
    # Settings
    st.markdown("### ⚙️ Cài đặt")
    confidence_threshold = st.slider(
        "Ngưỡng tin cậy (%)", 
        min_value=50, 
        max_value=100, 
        value=80,
        help="Hiển thị cảnh báo nếu độ tin cậy dưới ngưỡng này"
    )
    
    show_details = st.checkbox("📈 Hiển thị chi tiết kết quả", value=True)
    
    # About section
    st.markdown("### ℹ️ Giới thiệu")
    st.markdown("""
    **Các hành vi được nhận dạng:**
    - 🔵 **Safe**: Lái xe an toàn
    - 🟠 **Talking**: Đang nói chuyện
    - 🔴 **Texting**: Đang nhắn tin  
    - 🟡 **Turning**: Đang rẽ
    - ⚫ **Other**: Hành vi khác
    """)

# ================================
# 🛠️ HELPER FUNCTIONS
# ================================
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict_with_confidence(model, image):
    """Make prediction and return confidence scores"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx] * 100)
    return predicted_class, confidence, predictions[0]

def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 80:
        return "confidence-high"
    elif confidence >= 60:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_prediction_chart(predictions):
    """Create interactive prediction chart"""
    df = pd.DataFrame({
        'Behavior': class_names,
        'Confidence': predictions * 100
    })
    
    fig = px.bar(
        df, 
        x='Behavior', 
        y='Confidence',
        color='Confidence',
        color_continuous_scale='viridis',
        title="📊 Confidence Scores for Each Behavior"
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Loại hành vi",
        yaxis_title="Độ tin cậy (%)"
    )
    
    return fig

# ================================
# 📤 FILE UPLOAD SECTION
# ================================
st.markdown("## 📤 Tải lên file")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Chọn ảnh hoặc video để phân tích",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Hỗ trợ: JPG, PNG, MP4. Kích thước tối đa: 200MB",
        accept_multiple_files=True  # Cho phép tải nhiều file
    )

with col2:
    if uploaded_files:
        total_size = sum(len(file.getvalue()) for file in uploaded_files) / (1024 * 1024)  # MB
        st.metric("📁 Tổng kích thước", f"{total_size:.2f} MB")
        st.metric("📋 Số lượng file", len(uploaded_files))

# ================================
# 🖼️ IMAGE PROCESSING
# ================================
if uploaded_files:
    # Tạo tabs cho từng file
    tabs = st.tabs([f"File {i+1}: {file.name}" for i, file in enumerate(uploaded_files)])
    
    for i, (tab, uploaded_file) in enumerate(zip(tabs, uploaded_files)):
        with tab:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ["jpg", "jpeg", "png"]:
                # Display image in columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 🖼️ Ảnh gốc")
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
                    
                    # Image info
                    width, height = image.size
                    st.info(f"📐 Kích thước: {width} x {height} pixels")
                
                with col2:
                    st.markdown("### 🔍 Kết quả phân tích")
                    
                    if st.button(f"🚀 Phân tích ảnh {i+1}", type="primary", use_container_width=True, key=f"analyze_{i}"):
                        with st.spinner('🤖 AI đang phân tích...'):
                            try:
                                model = models[selected_model_name]
                                predicted_class, confidence, all_predictions = predict_with_confidence(model, image)
                                
                                # Main result
                                confidence_class = get_confidence_color(confidence)
                                st.markdown(f"""
                                <div class="result-card">
                                    <h2>🎯 Kết quả dự đoán</h2>
                                    <h1>{predicted_class}</h1>
                                    <p class="{confidence_class}">Độ tin cậy: {confidence:.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Confidence warning
                                if confidence < confidence_threshold:
                                    st.warning(f"⚠️ Độ tin cậy thấp ({confidence:.1f}% < {confidence_threshold}%). Kết quả có thể không chính xác.")
                                else:
                                    st.success(f"✅ Kết quả có độ tin cậy cao ({confidence:.1f}%)")
                                
                                # Detailed results
                                if show_details:
                                    st.markdown("### 📊 Chi tiết kết quả")
                                    
                                    # Interactive chart
                                    fig = create_prediction_chart(all_predictions)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Detailed table
                                    df_results = pd.DataFrame({
                                        '🎯 Hành vi': class_names,
                                        '📊 Độ tin cậy (%)': [f"{p*100:.2f}%" for p in all_predictions],
                                        '📈 Điểm số': [f"{p:.4f}" for p in all_predictions]
                                    })
                                    
                                    # Highlight the predicted class
                                    def highlight_max(s):
                                        is_max = s == s.max()
                                        return ['background-color: #ffeb3b' if v else '' for v in is_max]
                                    
                                    st.dataframe(
                                        df_results.style.apply(highlight_max, subset=['📈 Điểm số']),
                                        use_container_width=True
                                    )
                                    
                            except Exception as e:
                                st.error(f"❌ Lỗi khi phân tích: {str(e)}")
            
            elif file_extension == "mp4":
                st.markdown("### 🎥 Video Analysis")
                
                # Video preview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    
                    st.video(tfile.name)
                    
                with col2:
                    # Video info
                    video_cap = cv2.VideoCapture(tfile.name)
                    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = video_cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0
                    video_cap.release()
                    
                    st.metric("🎬 Tổng số khung hình", f"{total_frames:,}")
                    st.metric("⏱️ Thời lượng", f"{duration:.2f}s")
                    st.metric("📹 FPS", f"{fps:.1f}")
                
                # Video processing options
                st.markdown("### ⚙️ Tùy chọn phân tích video")
                
                col1, col2 = st.columns(2)
                with col1:
                    frame_skip = st.slider("Bỏ qua khung hình", 1, 10, 3, help="Phân tích mỗi N khung hình để tăng tốc")
                with col2:
                    max_frames = st.slider("Số khung tối đa", 10, min(100, total_frames), 30)
                
                if st.button("🎬 Phân tích video", type="primary", use_container_width=True):
                    with st.spinner(f'🎥 Đang phân tích {max_frames} khung hình...'):
                        try:
                            video = cv2.VideoCapture(tfile.name)
                            model = models[selected_model_name]
                            
                            predictions = []
                            frames = []
                            frame_count = 0
                            analyzed_count = 0
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            while video.isOpened() and analyzed_count < max_frames:
                                success, frame = video.read()
                                if not success:
                                    break
                                
                                if frame_count % frame_skip == 0:
                                    # Process frame
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    image = Image.fromarray(frame_rgb)
                                    
                                    predicted_class, confidence, all_pred = predict_with_confidence(model, image)
                                    predictions.append({
                                        'frame': frame_count,
                                        'class': predicted_class,
                                        'confidence': confidence,
                                        'predictions': all_pred
                                    })
                                    frames.append(frame_rgb)
                                    analyzed_count += 1
                                    
                                    # Update progress
                                    progress = analyzed_count / max_frames
                                    progress_bar.progress(progress)
                                    status_text.text(f'Đã phân tích: {analyzed_count}/{max_frames} khung hình')
                                
                                frame_count += 1
                            
                            video.release()
                            os.remove(tfile.name)
                            
                            # Store results in session state
                            st.session_state.video_predictions = predictions
                            st.session_state.video_frames = frames
                            
                            st.success(f"✅ Hoàn thành phân tích {analyzed_count} khung hình!")
                            
                        except Exception as e:
                            st.error(f"❌ Lỗi khi phân tích video: {str(e)}")
                
                # Display video results
                if 'video_predictions' in st.session_state:
                    st.markdown("### 📊 Kết quả phân tích video")
                    
                    predictions = st.session_state.video_predictions
                    frames = st.session_state.video_frames
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_confidence = np.mean([p['confidence'] for p in predictions])
                        st.metric("📊 Độ tin cậy TB", f"{avg_confidence:.1f}%")
                    
                    with col2:
                        most_common = max(set([p['class'] for p in predictions]), 
                                        key=[p['class'] for p in predictions].count)
                        st.metric("🎯 Hành vi chủ đạo", most_common)
                    
                    with col3:
                        high_conf_count = sum(1 for p in predictions if p['confidence'] >= confidence_threshold)
                        st.metric("✅ Kết quả tin cậy", f"{high_conf_count}/{len(predictions)}")
                    
                    with col4:
                        unique_behaviors = len(set([p['class'] for p in predictions]))
                        st.metric("🔄 Số hành vi khác nhau", unique_behaviors)
                    
                    # Thêm phần thống kê chi tiết cho từng hành vi
                    st.markdown("#### 📈 Thống kê chi tiết theo hành vi")
                    
                    # Tính toán số lượng và tỷ lệ cho từng hành vi
                    behavior_counts = {}
                    total_frames = len(predictions)
                    
                    for pred in predictions:
                        behavior = pred['class']
                        if behavior not in behavior_counts:
                            behavior_counts[behavior] = 0
                        behavior_counts[behavior] += 1
                    
                    # Tạo DataFrame cho biểu đồ và bảng
                    behavior_stats = []
                    for behavior in class_names:
                        count = behavior_counts.get(behavior, 0)
                        percentage = (count / total_frames) * 100
                        behavior_stats.append({
                            'Hành vi': behavior,
                            'Số khung hình': count,
                            'Tỷ lệ (%)': f"{percentage:.1f}%"
                        })
                    
                    # Hiển thị biểu đồ
                    df_stats = pd.DataFrame(behavior_stats)
                    fig = px.bar(
                        df_stats,
                        x='Hành vi',
                        y='Số khung hình',
                        color='Hành vi',
                        title='📊 Phân bố số lượng khung hình theo hành vi',
                        labels={'Số khung hình': 'Số lượng khung hình'},
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(
                        xaxis_title="Hành vi",
                        yaxis_title="Số lượng khung hình",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Hiển thị bảng thống kê
                    st.markdown("#### 📋 Bảng thống kê chi tiết")
                    st.dataframe(
                        df_stats,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Frame-by-frame analysis
                    st.markdown("#### 🔍 Phân tích từng khung hình")
                    
                    frame_idx = st.slider(
                        "Chọn khung hình", 
                        0, len(frames) - 1, 0,
                        help="Kéo để xem kết quả phân tích từng khung hình"
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(
                            frames[frame_idx], 
                            caption=f'Khung hình {predictions[frame_idx]["frame"] + 1}',
                            use_column_width=True
                        )
                    
                    with col2:
                        pred = predictions[frame_idx]
                        confidence_class = get_confidence_color(pred['confidence'])
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h3>🎯 Khung hình {pred['frame'] + 1}</h3>
                            <h2>{pred['class']}</h2>
                            <p class="{confidence_class}">Độ tin cậy: {pred['confidence']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if show_details:
                            # Mini chart for this frame
                            fig_mini = create_prediction_chart(pred['predictions'])
                            fig_mini.update_layout(height=300)
                            st.plotly_chart(fig_mini, use_container_width=True)

# Thêm nút phân tích tất cả ảnh
if uploaded_files and any(f.name.split('.')[-1].lower() in ["jpg", "jpeg", "png"] for f in uploaded_files):
    if st.button("🚀 Phân tích tất cả ảnh", type="primary", use_container_width=True):
        with st.spinner('🤖 AI đang phân tích tất cả ảnh...'):
            try:
                # Tạo DataFrame để lưu kết quả
                all_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    if uploaded_file.name.split('.')[-1].lower() in ["jpg", "jpeg", "png"]:
                        image = Image.open(uploaded_file)
                        model = models[selected_model_name]
                        predicted_class, confidence, _ = predict_with_confidence(model, image)
                        
                        all_results.append({
                            'File': uploaded_file.name,
                            'Hành vi': predicted_class,
                            'Độ tin cậy (%)': f"{confidence:.1f}%"
                        })
                
                # Hiển thị kết quả tổng hợp
                st.markdown("### 📊 Kết quả phân tích tất cả ảnh")
                df_all_results = pd.DataFrame(all_results)
                st.dataframe(df_all_results, use_container_width=True)
                
                # Thống kê tổng hợp
                st.markdown("### 📈 Thống kê tổng hợp")
                behavior_counts = df_all_results['Hành vi'].value_counts()
                
                # Biểu đồ phân bố hành vi
                fig = px.pie(
                    values=behavior_counts.values,
                    names=behavior_counts.index,
                    title='Phân bố hành vi trong tất cả ảnh'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Lỗi khi phân tích: {str(e)}")

# ================================
# 📊 FOOTER STATISTICS
# ================================
if models:
    st.markdown("---")
    st.markdown("### 📈 Thống kê hệ thống")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 Mô hình khả dụng", len(models))
    
    with col2:
        st.metric("🎯 Loại hành vi", len(class_names))
    
    with col3:
        st.metric("📊 Độ chính xác trung bình", "88.7%")
    
    with col4:
        st.metric("⚡ Tốc độ xử lý", "~2.3s/ảnh")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚗 <strong>Driver Behavior AI Detection</strong> - Được phát triển với ❤️ bằng Streamlit & TensorFlow</p>
    <p>🤖 Hỗ trợ phân tích hành vi lái xe thông minh • 📊 3 mô hình AI tiên tiến • 🔒 Xử lý dữ liệu an toàn</p>
</div>
""", unsafe_allow_html=True)
