# Mushroom Genus Classifier
Welcome to "Guess the Shroom!", a fun and smart web application that classifies mushrooms into one of 9 common genera based on their images! 🧠📸

<div align="center"> <img src="https://em-content.zobj.net/source/microsoft-teams/363/mushroom_1f344.png" width="100"/> </div>

### Project Overview
This project uses EfficientNetB0, a powerful image classification model, to identify the genus of a mushroom from a photo. It also includes a Flask-powered web app with a cute and interactive interface to make the classification experience more fun! 💖

### 🔍Features
- Deep learning-based image classification using EfficientNetB0
- Supports 9 mushroom genera: Agaricus, Amanita, Boletus, Cortinarius, Entoloma, Hygrocybe, Lactarius, Russula, Suillus
- Clean and interactive Flask web interface
- Automatically skips corrupted images
- Custom HTML/CSS frontend with a cute theme 🎨

### Tech Stack
1. Model Architecture	TensorFlow / Keras + EfficientNetB0
2. Dataset Source	KaggleHub - Common Mushroom Genera Images
3. Backend	Flask (Python)
4. Frontend	HTML + CSS

### How to Run Locally
Clone the repo:
```
git clone https://github.com/yourusername/mushroom-classifier.git
cd mushroom-classifier
```

Install dependencies:
```
pip install -r requirements.txt
Train the model (optional):
Use the script to train EfficientNetB0 with your dataset from Kaggle.
```

Run the web app:
```python app.py```

Open in browser:
Go to `http://127.0.0.1:5000` and try uploading your mushroom image!

### Dataset
Downloaded using KaggleHub from:
maysee/mushrooms-classification-common-genuss-images

Includes 9 mushroom genera with cleaned and resized image samples.

_Inspired by curiosity for 🍄 fungi and AI!_


