import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from urllib.request import urlopen
import os
import shutil

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
APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = 'vit_plantdisease.pt'
MODEL_PATH = APP_DIR / MODEL_FILENAME
MODEL_CACHE_PATH = Path.home() / '.cache' / 'crop-doctor' / MODEL_FILENAME
CLASSES_PATH = APP_DIR / 'classes.txt'
DEFAULT_MODEL_URL = 'https://huggingface.co/akshayjod/crop-doctor-model/resolve/main/vit_plantdisease.pt'


def get_config_value(*keys):
    for key in keys:
        try:
            value = st.secrets[key]
        except Exception:
            value = None
        if value:
            return value
        value = os.getenv(key)
        if value:
            return value
    return None


def download_model(download_url):
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_CACHE_PATH.with_suffix(MODEL_CACHE_PATH.suffix + '.download')
    try:
        with urlopen(download_url) as response, open(temporary_path, 'wb') as output_file:
            shutil.copyfileobj(response, output_file)
        temporary_path.replace(MODEL_CACHE_PATH)
        return MODEL_CACHE_PATH
    except Exception as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        raise FileNotFoundError(f"Unable to download model from {download_url}: {exc}") from exc


def resolve_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH

    download_url = get_config_value('MODEL_URL', 'CROP_DOCTOR_MODEL_URL', 'MODEL_DOWNLOAD_URL') or DEFAULT_MODEL_URL
    if download_url:
        return download_model(download_url)

    raise FileNotFoundError(
        f"Model file not found. Expected it at {MODEL_PATH}. "
        "For local runs, place vit_plantdisease.pt next to app.py. "
        "For Streamlit Community Cloud, the app will download the default Hugging Face model automatically, or you can override it with MODEL_URL or CROP_DOCTOR_MODEL_URL."
    )


def load_class_names():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Class label file not found at {CLASSES_PATH}.")
    return [line.strip() for line in CLASSES_PATH.read_text(encoding='utf-8').splitlines() if line.strip()]

try:
    class_names = load_class_names()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

@st.cache_resource
def load_model():
    import torchvision.models as models
    import torch.nn as nn
    num_classes = len(class_names)
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    for param in model.heads.head.parameters():
        param.requires_grad = True
    model_path = resolve_model_path()
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model weights from {model_path}: {exc}") from exc
    model.to(device)
    model.eval()
    return model

try:
    model = load_model()
except (FileNotFoundError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

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
