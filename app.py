from flask import Flask, render_template, request, url_for, redirect, send_from_directory
import os
from werkzeug.utils import secure_filename

from services.image_processing import process_images

import os
import threading
import webbrowser
import cv2


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index-home.html')

@app.route('/process_result', methods=['POST'])
def process_result():
    # Clean old uploaded images
    clean_folder(app.config['UPLOAD_FOLDER'])

    # Process uploaded images
    for i in range(1, 3):
        file = request.files.get(f'file{i}')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

    return redirect(url_for('index.result.html'))

def clean_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error cleaning {folder}: {e}")

if __name__ == '__main__':
    app.run(debug=True)
