from flask import Flask, render_template, request, flash, redirect, send_from_directory, url_for
import os

from services.image_processing import process_images
from services.mob_processing import process_images_mob
from services.web_processing import process_images_web

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def create_upload_folder():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

def create_processed_folder():
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)

create_upload_folder()
create_processed_folder()


def process_image(file_path):
    # Add your image processing logic here
    # You can use libraries like OpenCV, Pillow, etc.
    # Example: Resize the image
    from PIL import Image
    img = Image.open(file_path)
    img = img.resize((300, 300))
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + os.path.basename(file_path))
    img.save(processed_path)
    return processed_path

@app.route('/')
def index():
    return render_template('index-home.html')

@app.route('/upload/<int:image_number>', methods=['POST'])
def upload_image(image_number):
    create_upload_folder()

    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file:
        filename = f'image_{image_number}.jpg'  # file name can be changed
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('File uploaded successfully', 'success')

    return redirect(url_for('index'))


@app.route('/processed/<filename>')
def processed_image(filename):
    print('>>>>>>>')
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


@app.route('/process_result', methods=['POST', 'GET'])
def process_result():
    img_1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image_1.jpg')
    img_2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image_2.jpg')
    create_processed_folder()
    process_images_mob(img_1_path, img_2_path)

    processed_image_paths = {
        #'developer': process_image(os.path.join(app.config['UPLOAD_FOLDER'], 'image_2.jpg')),
        'invert': url_for('processed_image', filename='processed_image_1.jpg'),  # Use url_for to generate the URL
        'gray': url_for('processed_image', filename='processed_image_2.jpg')  # Use url_for to generate the URL
    }

    # Render the result page with the processed image paths
    return render_template('index-result.html')



if __name__ == '__main__':
    app.run(debug=True)
