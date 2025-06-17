from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model('shroom_cnn.h5')  # nama file model

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Baca label dari file
with open('class_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            img_input = image.load_img(img_path, target_size=(64, 64))  # sesuaikan dengan model kamu
            img_array = image.img_to_array(img_input)
            img_array = np.expand_dims(img_array, axis=0) / 255.

            pred = model.predict(img_array)
            label_index = np.argmax(pred)
            prediction = labels[label_index]
            image_url = '/' + img_path

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
