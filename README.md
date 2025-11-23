# 🎭 Human Face Emotion Classification

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

## 📋 Overview
Proyek ini adalah model **Deep Learning** berbasis **Convolutional Neural Network (CNN)** untuk mengklasifikasikan emosi wajah manusia ke dalam tiga kategori utama: **Positive, Negative, dan Neutral**.

Model ini dikembangkan menggunakan **TensorFlow/Keras** dengan fokus pada optimasi pipeline data dan deployment multi-platform (Mobile & Web).

## 🚀 Key Features
* **Data Pipeline Efisien**: Menggunakan `tf.data` dengan caching dan prefetching untuk memaksimalkan penggunaan GPU.
* **Robust Training**: Menerapkan strategi `Class Weights` untuk menangani ketidakseimbangan data, serta Callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`).
* **Clean Inference Architecture**: Memisahkan layer augmentasi data dari model final untuk memastikan kompatibilitas penuh saat ekspor.
* **Multi-Platform Export**: Model tersedia dalam format:
    * `SavedModel` (Standar TensorFlow)
    * `TF-Lite` (Android/iOS/IoT)
    * `TFJS` (Browser/Web App)

## 📊 Dataset
Dataset asli diambil dari [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) dan diproses ulang menjadi 3 kelas:
1.  **Positive**: (Happy)
2.  **Negative**: (Angry, Fear, Sad)
3.  **Neutral**: (Surprise/Neutral)

*Preprocessing*: Dataset dibagi secara fisik menggunakan `split-folders` menjadi Train (80%), Validation (19%), dan Test (1%).

## 🛠️ Model Architecture
Arsitektur model menggunakan Sequential API dengan urutan:
1.  **Input Layer**: (48, 48, 1) Grayscale.
2.  **Data Augmentation**: RandomFlip & RandomRotation (Hanya saat training).
3.  **Feature Extraction**: 4 Blok Convolutional (Conv2D + BatchNormalization + MaxPooling2D).
4.  **Classifier**: Flatten + Dense (ReLU) + Dropout (0.5) + Output (Softmax).

## 📈 Results
Model dievaluasi menggunakan data Test yang tidak pernah dilihat sebelumnya (Unseen Data).
* **Test Accuracy**: ~90% (Mencapai target >85%)
* **Loss Function**: Sparse Categorical Crossentropy
* **Optimizer**: Adam

## 📂 File Structure
