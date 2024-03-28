from flask import Flask, render_template, request, url_for, redirect, send_from_directory
import os
from werkzeug.utils import secure_filename

from services.image_processing import process_images

import os
import threading
import webbrowser
import cv2

app = Flask(__name__)

# Placeholder variables to store uploaded image paths
#design_team_image_path = None
#developer_team_image_path = None

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def process_images_thread(design_image_path, developer_image_path):
    processed_images = process_images(design_image_path, developer_image_path)
    
    # Save processed images
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], 'processed_design.png'), processed_images[0])
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], 'processed_developer.png'), processed_images[1])

    # Open result page in a new tab
    webbrowser.open_new_tab(url_for('show_result'))

@app.route('/')
def index_home():
    return render_template('index-home.html')

@app.route('/upload/<int:image_number>', methods=['POST'])
def upload_image(image_number):
    file = request.files['file']

    if file:
        # Save the uploaded image
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'uploaded_image_{image_number}.png'))

    return redirect(url_for('index_home'))

@app.route('/process_result', methods=['POST'])
def process_result():
    # Start a new thread for image processing
    design_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image_1.png')
    developer_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image_2.png')
    
    processing_thread = threading.Thread(target=process_images_thread, args=(design_image_path, developer_image_path))
    processing_thread.start()

    return render_template('processing.html')

@app.route('/show_result')
def show_result():
    # Load processed images
    processed_design_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_design.png')
    processed_developer_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_developer.png')

    return render_template('index-result.html', processed_design=processed_design_path, processed_developer=processed_developer_path)

"""
@app.route('/favicon.ico')
def favicon():
    #print("favicon.ico Locantion is not avialable, look method favicon() and location")
    return send_from_directory(
        app.root_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon'
    )
"""

if __name__ == "__main__":
    app.run(debug=True)