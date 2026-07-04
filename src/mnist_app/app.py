import base64
import io
import os
from pathlib import Path
import base64
import io
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
MODEL_PATH = BASE_DIR / 'model.h5'
HISTORY_DIR = ROOT_DIR / 'static' / 'assets' / 'histories'
ALLOWED_UPLOAD_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = tf.keras.models.load_model(str(MODEL_PATH))

app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / 'templates'),
    static_folder=str(ROOT_DIR / 'static'),
)


def ensure_history_dir() -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR


def next_history_path(history_dir: Path, prefix: str) -> Path:
    index = 0
    while True:
        image_path = history_dir / f'{prefix}_{index}.png'
        if not image_path.exists():
            return image_path
        index += 1


def decode_image_from_data_url(data_url: str) -> Image.Image:
    _, encoded = data_url.split(',', 1)
    image_bytes = io.BytesIO(base64.b64decode(encoded))
    return Image.open(image_bytes).convert('L')


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = gaussian_filter(arr, sigma=0.5)
    return arr.reshape(1, 28, 28, 1)


def save_image(image_array: np.ndarray, prefix: str) -> str:
    history_dir = ensure_history_dir()
    image_path = next_history_path(history_dir, prefix)
    plt.imsave(image_path, image_array.reshape(28, 28), cmap='gray')
    return str(Path('assets') / 'histories' / image_path.name)


@app.route('/')
def index() -> Any:
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload() -> Any:
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded or selected'}), 400

    filename = secure_filename(file.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        return jsonify({'error': 'Only PNG, JPG, JPEG, and WEBP images are supported'}), 400

    history_dir = ensure_history_dir()
    upload_path = next_history_path(history_dir, 'upload')

    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({'error': 'The uploaded file is not a valid image'}), 400

    image.save(upload_path)
    return jsonify({'image_path': str(Path('assets') / 'histories' / upload_path.name)})


@app.route('/predict', methods=['POST'])
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    data_url = payload.get('image')
    if not data_url:
        return jsonify({'error': 'No image data found'}), 400

    image = decode_image_from_data_url(data_url)
    processed = preprocess_image(image)
    save_image(processed, 'processed')

    prediction = model.predict(processed, verbose=0)
    digit = int(np.argmax(prediction, axis=1)[0])
    probabilities = [float(p) for p in prediction[0]]

    return jsonify({'digit': digit, 'probabilities': probabilities})


@app.route('/update_model', methods=['POST'])
def update_model() -> Any:
    payload = request.get_json(silent=True) or {}
    repo_id = payload.get('repo_id')
    
    if not repo_id:
        return jsonify({'error': 'Please provide Hugging Face Repo ID'}), 400

    try:
        # Tải file model.h5 từ Hugging Face
        downloaded_path = hf_hub_download(repo_id=repo_id, filename="model.h5")
        
        # Ghi đè file model.h5 cục bộ bằng shutil
        import shutil
        shutil.copy(downloaded_path, str(MODEL_PATH))
        
        # Nạp lại model vào bộ nhớ (Global update)
        global model
        model = tf.keras.models.load_model(str(MODEL_PATH))
        
        return jsonify({'message': f'Successfully loaded model from {repo_id}'})
    except Exception as e:
        return jsonify({'error': f'Error loading model: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
