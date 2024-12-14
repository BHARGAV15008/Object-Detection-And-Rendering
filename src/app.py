import os
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from utils import unzip_dataset
from train import train_model
from detect import detect_objects

# Configuration
UPLOAD_FOLDER = '../data/raw'
PROCESSED_FOLDER = '../data/processed'
MODEL_FOLDER = '../data/models'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        unzip_dataset(filepath, PROCESSED_FOLDER)
        model = train_model(PROCESSED_FOLDER, os.path.join(MODEL_FOLDER, 'object_detector.h5'))
        
        return 'Training Complete! Model saved.'

@app.route('/detect', methods=['GET'])
def detect():
    model_path = os.path.join(MODEL_FOLDER, 'object_detector.h5')
    model = load_model(model_path)
    detect_objects(model, live_stream=True)
    return 'Detection Started! Press Q to stop.'

if __name__ == '__main__':
    app.run(debug=True)
