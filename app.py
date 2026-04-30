from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model/plant_model.keras")

# =========================
# AUTO LOAD CLASS NAMES
# =========================
class_names = sorted(os.listdir("dataset/train"))

# =========================
# UPLOAD FOLDER
# =========================
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template('index.html')


# =========================
# PREDICTION FUNCTION
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return "❌ No file uploaded"

    file = request.files['image']

    if file.filename == '':
        return "❌ No selected file"

    # Save file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Read and preprocess image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.reshape(img, (1, 128, 128, 3))

        # Predict
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        result = class_names[class_index]

    except Exception as e:
        return f"❌ Error: {str(e)}"

    return render_template(
        'result.html',
        prediction=result,
        confidence=round(confidence * 100, 2),
        image_path=filepath
    )


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)