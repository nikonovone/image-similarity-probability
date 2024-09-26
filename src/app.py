import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

root_dir = Path("./")
sys.path.append(root_dir.resolve().as_posix())

from src.models import CompareModel


CHECKPOINT_PATH = Path("../logs/version_0/model.ckpt").as_posix()


# Load trained model
@st.cache(allow_output_mutation=True)
def load_model(checkpoint_path, device):
    model = CompareModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)  # Move model to the selected device
    return model


# Preprocess image for the model
def preprocess_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ],
    )
    return preprocess(image).unsqueeze(0)  # Add batch dimension


# Calculate similarity between embeddings
def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()


# Streamlit App
st.title("Image Similarity Checker")

# Set directory for model checkpoints and device
device_option = st.selectbox("Choose a device", ("cuda", "cpu"))
device = torch.device(device_option if torch.cuda.is_available() else "cpu")


model = load_model(CHECKPOINT_PATH, device)

if model:
    # Upload two images for similarity check
    st.write("Upload two images to check their similarity:")
    uploaded_file1 = st.file_uploader(
        "Choose the first image",
        type=["png", "jpg", "jpeg"],
    )
    uploaded_file2 = st.file_uploader(
        "Choose the second image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file1 and uploaded_file2:
        # Load the uploaded images
        image1 = Image.open(uploaded_file1).convert("RGB")
        image2 = Image.open(uploaded_file2).convert("RGB")

        # Display images
        st.image([image1, image2], caption=["First Image", "Second Image"], width=300)

        # Preprocess the images
        img1_tensor = preprocess_image(image1).to(device)
        img2_tensor = preprocess_image(image2).to(device)

        # Get embeddings from the model
        with torch.no_grad():
            embedding1 = model(img1_tensor)
            embedding2 = model(img2_tensor)

        # Calculate similarity
        similarity = cosine_similarity(embedding1, embedding2)

        # Display similarity result
        st.write(f"Similarity Score (Cosine): {similarity:.4f}")

        # Visualize embeddings (optional - you can skip this if not needed)
        embeddings_grid = make_grid(
            torch.cat([img1_tensor.cpu(), img2_tensor.cpu()], dim=0),
        )  # Convert to CPU for visualization
        st.image(
            embeddings_grid.permute(1, 2, 0).numpy(),
            caption="Image Embeddings",
            width=600,
        )
