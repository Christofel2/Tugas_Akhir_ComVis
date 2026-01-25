# Klasifikasi Tumor Otak Menggunakan ResNet50 & Streamlit

Proyek ini adalah solusi *End-to-End Computer Vision* untuk mengklasifikasikan jenis tumor otak dari citra MRI. Proyek ini mencakup tahapan persiapan data, pelatihan model *Deep Learning* menggunakan arsitektur **ResNet50** (*Transfer Learning*), dan *deployment* model secara offline menggunakan aplikasi web berbasis **Streamlit**.

## 🔍 Gambaran Umum

Sistem ini dirancang untuk mendeteksi dan mengklasifikasikan citra MRI otak ke dalam 4 kategori diagnosis:

1. **Glioma**
2. **Meningioma**
3. **Pituitary** (Tumor Kelenjar Hipofisis)
4. **No Tumor** (Normal)

Aplikasi ini memungkinkan pengguna untuk mengunggah gambar MRI melalui antarmuka web sederhana dan mendapatkan hasil prediksi secara *real-time*.

---

## 📂 Struktur File

Proyek ini terdiri dari tiga komponen kode utama:

1. **`Data_Processing_ComVis.ipynb`**
* **Fungsi:** Mengunduh dataset dari Kaggle, mengekstrak file, membagi data (Train/Val/Test), dan mengemasnya kembali agar siap dilatih.


2. **`Resnet50_Model_ComVis.ipynb`**
* **Fungsi:** Membangun model *Deep Learning*, melakukan *Data Augmentation*, melatih model (Training), dan mengevaluasi akurasi. Notebook ini menghasilkan file model `brain_tumor_final.h5`.


3. **`app.py`**
* **Fungsi:** Aplikasi antarmuka pengguna (GUI) berbasis Streamlit untuk memuat model `.h5` dan melakukan prediksi pada gambar yang diunggah pengguna.



---

## 🧠 Dataset

Data yang digunakan bersumber dari Kaggle: **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)**.

* **Input:** Citra MRI (JPG/PNG).
* **Preprocessing:** Resize ke 224x224 piksel, normalisasi nilai piksel sesuai standar ResNet50.

---

## 🛠 Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3.x
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (cv2), Pillow (PIL)
* **Data Manipulation:** Pandas, NumPy, Shutil
* **Deployment / UI:** Streamlit
* **Environment:** Google Colab (untuk Training) & Local Machine (untuk Deployment)

---

## 🏗 Arsitektur Model

Model dibangun menggunakan teknik **Transfer Learning**:

1. **Base Model:** ResNet50 (Pre-trained on ImageNet).
* *Layers Frozen:* Layer awal dibekukan agar bobot fitur dasar tidak rusak.


2. **Custom Head (Top Layers):**
* Global Average Pooling 2D
* Dense Layer (256 neuron, ReLU) + Dropout (0.3)
* Dense Layer (128 neuron, ReLU) + Dropout (0.2)
* Output Layer (4 neuron, Softmax)


3. **Optimizer:** Adam (Learning Rate = 0.001 dengan *scheduler*).

---

## 📊 Hasil Evaluasi

Berdasarkan proses pelatihan terakhir, model mencapai performa sebagai berikut:

* **Akurasi Training:** ~94.27%
* **Akurasi Validasi:** ~94.29%
* **Konvergensi:** Dicapai dalam kurang lebih 15 epoch.

---

## 🚀 Instalasi & Cara Menjalankan

Ikuti langkah-langkah berikut untuk menjalankan proyek ini dari awal hingga aplikasi siap digunakan.

### 1. Persiapan Lingkungan (Environment)

Pastikan Python sudah terinstal, lalu instal *library* yang dibutuhkan:

```bash
pip install tensorflow streamlit pandas numpy matplotlib pillow kaggle

```

### 2. Pengolahan Data & Pelatihan Model

Langkah ini disarankan dijalankan di **Google Colab** atau mesin dengan GPU.

1. Buka dan jalankan **`Data_Processing_ComVis.ipynb`**.
* Pastikan kamu memiliki file `kaggle.json` (API Token) untuk mengunduh dataset.


2. Buka dan jalankan **`Resnet50_Model_ComVis.ipynb`**.
* Notebook ini akan melatih model menggunakan data yang sudah diproses.
* **PENTING:** Setelah pelatihan selesai, unduh file model yang dihasilkan (biasanya bernama `brain_tumor_final.h5`) ke komputer lokal kamu.



### 3. Menjalankan Aplikasi Streamlit (Deployment)

Setelah memiliki file model (`brain_tumor_final.h5`), kamu bisa menjalankan aplikasi secara offline.

1. Letakkan file berikut dalam satu folder yang sama:
* `app.py`
* `brain_tumor_final.h5`


2. Buka terminal/command prompt, arahkan ke folder tersebut, dan jalankan perintah:
```bash
streamlit run app.py

```


3. Browser akan otomatis terbuka (biasanya di `http://localhost:8501`).
4. Unggah gambar MRI otak dan lihat hasil prediksinya!

---

*Dibuat untuk tugas Computer Vision.*
