from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import base64
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

#* Flask: Python web framework
#   - Input: request 
#   - Output: response (jsonify)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

Directory = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(Directory, 'template'),
            static_folder=os.path.join(Directory, 'static'))

model = tf.keras.models.load_model(os.path.join(Directory, 'source', 'model.h5'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or not request.files['file']:
        return jsonify({'error': 'Không có file được tải lên'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    histories_folder = os.path.join(Directory, 'static', 'assets', 'histories')
    os.makedirs(histories_folder, exist_ok=True)
    image_path = os.path.join(histories_folder, f'upload_{len(os.listdir(histories_folder))}.png')
    file.save(image_path)
    relative_image_path = os.path.join('assets', 'histories', os.path.basename(image_path))
    return jsonify({'image_path': relative_image_path})

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.json['image']
    header, encoded = data_url.split(',', 1)
    img_bytes = io.BytesIO(base64.b64decode(encoded))
    image = Image.open(img_bytes).convert('L')

    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(image).astype('float32') / 255.0
    arr = gaussian_filter(arr, sigma=0.5)  # Giảm sigma để giữ chi tiết
    arr = arr.reshape(1, 28, 28, 1)

    histories_folder = os.path.join(Directory, 'static', 'assets', 'histories')
    os.makedirs(histories_folder, exist_ok=True)
    plt.imsave(os.path.join(histories_folder, f'processed_{len(os.listdir(histories_folder))}.png'), arr.reshape(28, 28), cmap='gray')
    
    prediction = model.predict(arr, verbose=0)
    digit = int(np.argmax(prediction, axis=1)[0])
    probabilities = [float(p) for p in prediction[0]]  # Chuyển xác suất thành danh sách

    return jsonify({'digit': digit, 'probabilities': probabilities})

if __name__ == '__main__':
    app.run(debug=True)