import streamlit as st
import numpy as np
import cv2
from PIL import Image

def clahe_enhance(image_array, clip_limit=2.0, tile_grid_size=(8, 8)):
    try:
        # Konversi ke grayscale
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # Terapkan CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_img = clahe.apply(img_gray)

        return enhanced_img

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Streamlit app
def main():
    st.title("Coba CLAHE")
    st.write("Preprocessing Gambar Fundus dengan CLAHE")

    uploaded_file = st.file_uploader("Unggah gambar fundus", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar dan konversi ke RGB
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Konversi ke numpy array
        image_array = np.array(image)
        
        # Terapkan CLAHE
        preprocessed_image = clahe_enhance(image_array)

        if preprocessed_image is not None:
            st.image(preprocessed_image, caption="Hasil CLAHE", use_column_width=True, clamp=True)

if __name__ == "__main__":
    main()
