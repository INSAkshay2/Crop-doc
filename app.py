import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Set Streamlit page configuration for aesthetics
st.set_page_config(
    page_title="Crop Doctor",
    page_icon="🌿",
    layout="wide"
)

# Custom theme styling (this can also be placed in .streamlit/config.toml if deploying)
st.markdown(
    """
    <style>
    .main {background-color: #F5F5F5;}
    .stButton>button {background-color:#4CAF50; color:white;}
    .stFileUploader {border: 2px solid #4CAF50;}
    </style>
    """,
    unsafe_allow_html=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names for predictions
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry_healthy', "Cherry(including_sour)Powdery_mildew", "Cherry(including_sour)healthy",
    'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight',
    'Corn(maize)healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy',
    'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight',
    'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch',
    'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite',
    'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy'
]

@st.cache_resource
def load_model():
    import torchvision.models as models
    import torch.nn as nn
    num_classes = len(class_names)
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    for param in model.heads.head.parameters():
        param.requires_grad = True
    model.load_state_dict(torch.load('vit_plantdisease.pt', map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Sidebar with instructions
with st.sidebar:
    st.title("🌾 Crop Doctor")
    st.info("""
        **How to use:**
        - Upload a clear, close-up image of a plant leaf.
        - Wait for the model to analyze and predict the disease (or health).
        - Results and help appear on this page instantly!

        ---
        _AI disease prediction for farmers and researchers._
    """)

# Main title and subtitle
st.markdown("<h1 style='color:#388e3c;'>🌱 Crop Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#666'>Empowering Farmers with AI — by Akshay</h4>", unsafe_allow_html=True)
st.markdown("""
<div style='background-color:#E8F5E9; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
    Upload a <b>clear image of a plant leaf</b> below to <span style='color:#388e3c;'>get a diagnosis with a single click!</span>
</div>
""", unsafe_allow_html=True)

# Feature: file uploader
uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=['jpg', 'jpeg', 'png'])

# Columns for image/result display and details
col1, col2 = st.columns([1, 2])
with col1:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True, output_format="PNG")
with col2:
    if uploaded_file:
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            result = class_names[predicted.item()]
        # Animated/confetti feedback
        if "healthy" in result.lower():
            st.success(f"🌟 **Prediction:** {result} (Your plant is healthy!)")
            st.balloons()
        else:
            st.error(f"⚠️ **Prediction:** {result} (A disease was detected!)")
            st.markdown("""
            <span style='color:#c62828;'>It's advised to take action—search remedies for this disease or consult an expert.</span>
            """, unsafe_allow_html=True)

# Expander for extra info
with st.expander("ℹ️ How does this app work?"):
    st.markdown("""
        - This app uses a Vision Transformer (ViT) deep learning model trained on 38 different plant diseases and healthy categories.
        - Input images are resized, normalized, and analyzed automatically for best prediction accuracy.
        - No user data is stored—everything runs locally in your session or securely on Streamlit Cloud.
    """)

# Footer
st.markdown("<hr style='border:2px solid #4CAF50; margin-top:40px;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:gray;'>Made by AKSHAY &nbsp; | &nbsp; Powered by Streamlit</div>",
    unsafe_allow_html=True
)
