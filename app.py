import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np

# Load model
model = load_model("brain_tumor_final.h5")

# Class order HARUS sama seperti saat training
class_names = ['Glioma', 'Meningioma', 'No_tumor', 'Pituitary']

st.title("Brain Tumor Classification (MRI)")

# --- Inisialisasi session state untuk menyimpan history ---
if "history" not in st.session_state:
    st.session_state.history = []   # berisi list tuple (image, pred_label, raw_softmax)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

if uploaded_file is not None:
    # --- Load image ---
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))

    # --- Convert ke array ---
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # --- Predict ---
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]

    # --- Simpan ke history ---
    st.session_state.history.append((img.copy(), predicted_class, prediction))

# --- Tampilkan semua history di bawah ---
st.write("## Prediction History")

if len(st.session_state.history) == 0:
    st.info("Belum ada gambar yang diprediksi.")
else:
    for i, (img, pred_label, raw_softmax) in enumerate(st.session_state.history):
        st.write(f"### Image {i+1}")
        st.image(img, width=250)
        st.write(f"**Prediction:** {pred_label}")
        st.write("Raw softmax:", raw_softmax)
        st.write("---")
