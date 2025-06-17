# Mushroom Genus Classifier
Welcome to "Shroomie Showdown!", a fun and smart web application that classifies mushrooms into one of 9 common genera based on their images! 

<div align="center"> 

<img src="https://github.com/user-attachments/assets/6bc744a0-640e-4665-a79b-2d10ef7193ac" alt="Alt Text" width="400" height="400">

</div>

## Project Overview
This project uses CNN, a powerful image classification model, to identify the genus of a mushroom from a photo. It also includes a Flask-powered web app with a cute and interactive interface to make the classification experience more fun! 

### Tech Stack
1. Model Architecture	TensorFlow / Keras 
2. Dataset Source	KaggleHub (Common Mushroom Genera Images) 
3. Backend	Flask (Python)
5. Frontend	HTML + CSS

## CNN Implementation

### Konsep umum
Convolutional Neural Network (CNN) adalah salah satu jenis algoritma deep learning yang paling efektif untuk pengenalan gambar. CNN bekerja dengan cara mengekstrak fitur dari gambar melalui operasi konvolusi, lalu mengklasifikasikan gambar berdasarkan fitur-fitur tersebut.

Struktur umum CNN terdiri dari:
1. Convolution Layer: Mendeteksi fitur dasar dari gambar seperti tepi, sudut, warna, dll.
2. Pooling Layer: Mengurangi dimensi data sambil mempertahankan fitur penting (biasanya dengan MaxPooling).
3. Flatten Layer: Mengubah hasil konvolusi 2D menjadi vektor 1D.
4. Fully Connected Layer (Dense): Mempelajari hubungan kompleks antara fitur dan melakukan klasifikasi akhir.
5. Dropout Layer: Digunakan untuk mengurangi overfitting dengan menonaktifkan neuron secara acak selama pelatihan.

### CNN in Mushroom Recognition
#### 1. Import library dan setup
```
import os
import shutil
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

```
- Melakukan import library untuk operasi file (os, shutil), pemrosesan gambar (PIL), dan array numerik (numpy).
- Baris ImageFile.LOAD_TRUNCATED_IMAGES = True memastikan gambar yang tidak utuh tetap bisa diproses.

#### 2. Import modul deep learning
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
```
Mengimpor komponen dari TensorFlow untuk membangun dan melatih model CNN.

#### 3. Mount google drive dan setup data augmentation
```
from google.colab import drive
drive.mount('/content/drive')

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
```
- Mengakses dataset jamur yang disimpan di Google Drive.
- `ImageDataGenerator` digunakan untuk augmentasi data dan pembagian data menjadi training dan validation.
- `rescale=1./255` berarti normalisasi pixel gambar dari 0-255 menjadi 0-1.

#### 4. Load dataset gambar
```
train_data = datagen.flow_from_directory(
    '/content/drive/MyDrive/Mushrooms',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    '/content/drive/MyDrive/Mushrooms',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

- Gambar akan otomatis dikategorikan berdasarkan subfolder nama kelas (misalnya: Mushrooms/Amanita, Mushrooms/Boletus, dll).
- Ukuran gambar dikonversi menjadi 64x64 piksel.
- `class_mode='categorical'` digunakan untuk klasifikasi multi-kelas.

#### 5. Membangun arsitektur CNN
```
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(train_data.class_indices), activation='softmax')
])
```

- Conv2D(32, (3,3)): Layer konvolusi pertama, menghasilkan 32 filter berukuran 3x3.
- MaxPooling2D(2,2): Mengurangi ukuran citra sebanyak setengahnya.
- Conv2D(64, (3,3)): Layer konvolusi kedua, menambah kompleksitas fitur.
- Flatten(): Mengubah hasil konvolusi menjadi array 1 dimensi.
- Dense(128): Layer fully-connected dengan 128 neuron.
- Dropout(0.3): Mengurangi overfitting dengan menonaktifkan 30% neuron saat training.
- Dense(..., activation='softmax'): Layer output dengan jumlah neuron sesuai jumlah kelas, menghasilkan probabilitas untuk tiap kelas.

#### 6. Kompilasi model
```
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
- Menggunakan Adam optimizer untuk pembaruan bobot yang efisien.
- `categorical_crossentropy` digunakan karena ini adalah kasus klasifikasi multi-kelas.
- `metrics=['accuracy']:` Untuk memonitor akurasi saat training.

#### 7. Training model
```
model.fit(train_data, epochs=10, validation_data=val_data)
```
Melatih model selama 10 epoch menggunakan data latih dan validasi.

![image](https://github.com/user-attachments/assets/379e4879-fe5a-4bfe-baae-762002ad5460)


#### 8. Menyimpan model dan label kelas
```
model.save('mushroom_cnn.h5')

with open('class_labels.txt', 'w') as f:
    for label in train_data.class_indices:
        f.write(f"{label}\n")
```
- Model yang telah dilatih disimpan sebagai file .h5 agar dapat digunakan untuk prediksi di masa depan.
- Label kelas disimpan ke dalam file teks class_labels.txt.


## How to Run Locally
Clone the repo:
```
git clone https://github.com/yourusername/mushroom-classifier.git
cd mushroom-classifier
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the web app:
```python app.py```

Open in browser:
Go to `http://127.0.0.1:5000` and try uploading your mushroom image!

## Documentation
1. Akses melalui localhost

![image](https://github.com/user-attachments/assets/30244dc0-5b3e-4a83-ab62-8dd89f2f4505)

2. Upload gambar jamur lalu klik tombol identifikasi
   
![image](https://github.com/user-attachments/assets/7f7a0e22-9cee-4860-81a7-db471c25de56)

3. Program akan menampilkan prediksi genus jamur yang paling mendekati

## Dataset
Downloaded using KaggleHub from:
maysee/mushrooms-classification-common-genuss-images

Includes 9 mushroom genera with cleaned and resized image samples.



_Inspired by curiosity for 🍄 fungi and AI!_


