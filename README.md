# Plant Disease Detection

A Flask web application that detects plant disease in uploaded leaf images using a trained TensorFlow/Keras model.

## Project Structure

The project is organized as follows:

```text
plant-disease-detection/
├── app.py
├── train.py
├── requirements.txt
├── model/
│   └── plant_model.keras
├── dataset/
│   ├── train/
│   │   ├── Tomato_Early_blight/
│   │   └── Tomato_healthy/
│   └── test/
│       ├── Tomato_Early_blight/
│       └── Tomato_healthy/
├── static/
│   ├── style.css
│   └── uploads/
└── templates/
    ├── index.html
    └── result.html
```

- `app.py` - Flask application entrypoint and prediction logic.
- `train.py` - (Optional) training script for model development.
- `requirements.txt` - Python dependencies.
- `model/plant_model.keras` - Trained Keras model used for inference.
- `dataset/` - Training and testing datasets organized by class.
  - `dataset/train/` contains class folders used to build the label list.
  - `dataset/test/` contains test images.
- `static/` - Static assets including CSS and uploaded images.
- `templates/` - HTML templates for the homepage and result page.

## Setup

1. Create and activate a Python virtual environment.

   Windows PowerShell:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

> Note: The `requirements.txt` file lists `tensorflow-macos`, which is intended for macOS systems. On Windows, install `tensorflow` instead:
>
> ```bash
> pip install tensorflow
> ```

## Running the App

Run the Flask application from the project root:

```bash
python app.py
```

Then open the browser to:

```
http://127.0.0.1:5000
```

## How It Works

- The app loads a trained model from `model/plant_model.keras`.
- It automatically loads class names from `dataset/train/`.
- Uploaded images are resized to `128x128`, normalized, and passed to the model.
- The predicted class and confidence score are shown on the result page.

## Notes

- Uploaded images are saved to `static/uploads/`.
- The app is currently configured for development mode with `debug=True`.
- If you want to retrain the model or prepare more data, use the `train.py` script.

## Dependencies

- Flask
- TensorFlow / TensorFlow-macos
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- Matplotlib

## License

This project does not include a license file. Add one if you want to share or publish the code.
