import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

def clahe_enhance(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
   
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_img = clahe.apply(img)
        return enhanced_img

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Streamlit app
def main():
    st.title("coba clahe")
    st.write("preprop.")

    uploaded_file = st.file_uploader("Unggah gambar fundus", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar yang diunggah
        image = Image.open(uploaded_file).convert("RGB")  # Pastikan gambar selalu RGB
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        
        # Preprocessing gambar
        image_array = np.array(image)
        preprocessed_image = clahe_enhance(image_array)

if __name__ == "__main__":
    main()
