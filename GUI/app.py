from flask import Flask, request, render_template
import os
import numpy as np
from glob import glob
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename
import shutil
import joblib

app = Flask(__name__)

IMG_SIZE = (160, 160)  # Updated to match the working model
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the working model
model = load_model('efficientnet_aln_model_combined1.keras', compile=False)
print("Model loaded successfully!")
print(f"Model input shape: {model.input_shape}")

scaler = joblib.load('scaler.save')
class_names = ['N+(1-2)', 'N+(>2)', 'N0']  # Fixed label encoding order


def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Ensure 3 channels (RGB)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)  # Ensure float32 type
    img = preprocess_input(img)
    return img


def predict_with_visuals(folder_path, age, tumor_size, ki67, max_images=25):
    image_paths = glob(os.path.join(folder_path, '*.jpg'))
    image_paths = random.sample(image_paths, min(len(image_paths), max_images))

    if not image_paths:
        return "No images found.", [], [], []

    images = np.array([preprocess_image(p).numpy() for p in image_paths])
    clinical_data = np.array([[age, tumor_size, ki67]] * len(images), dtype=np.float32)
    clinical_data = scaler.transform(clinical_data)

    preds = model.predict([images, clinical_data])
    pred_classes = np.argmax(preds, axis=1)

    # Majority vote
    majority_class = np.bincount(pred_classes).argmax()
    aln_status = class_names[majority_class]

    # Calculate average probabilities for bar graph
    avg_probabilities = np.mean(preds, axis=0)
    prob_data = []
    for i, class_name in enumerate(class_names):
        prob_data.append({
            'class': class_name,
            'probability': float(avg_probabilities[i] * 100)  # Convert to percentage
        })

    # Image-label pairs with names
    image_preds = []
    image_names_with_classes = []
    for path, pred in zip(image_paths, pred_classes):
        rel_path = os.path.relpath(path, start='static')  # for web access
        image_name = os.path.basename(path)  # Get just the filename
        image_preds.append((f'/static/{rel_path}', class_names[pred]))
        image_names_with_classes.append({
            'name': image_name,
            'class': class_names[pred],
            'class_index': pred
        })

    return aln_status, image_preds, prob_data, image_names_with_classes


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = float(request.form['age'])
        tumor_size = float(request.form['tumor_size'])
        ki67 = float(request.form['ki67'])
        files = request.files.getlist("images")

        patient_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        if os.path.exists(patient_folder):
            shutil.rmtree(patient_folder)
        os.makedirs(patient_folder)

        for file in files:
            if file and file.filename.endswith('.jpg'):
                filename = secure_filename(file.filename.split('/')[-1])  # remove folder path
                file.save(os.path.join(patient_folder, filename))

        prediction, image_preds, prob_data, image_names_with_classes = predict_with_visuals(patient_folder, age, tumor_size, ki67)
        return render_template('result.html', prediction=prediction, image_preds=image_preds, prob_data=prob_data, image_names_with_classes=image_names_with_classes)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
