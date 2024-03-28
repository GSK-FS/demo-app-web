from flask import Flask, render_template, request, redirect
import os

app = Flask(__name__)

# Set the upload and processed folders
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_'+UPLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # Check if the post request has the file parts
    if 'image1' not in request.files or 'image2' not in request.files:
        return redirect(request.url)

    file1 = request.files['image1']
    file2 = request.files['image2']

    # If the user does not select a file, the browser may send an empty file
    if file1.filename == '' or file2.filename == '':
        return redirect(request.url)

    # Ensure the 'uploads' and 'processed' folders exist
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Save the files to the uploads folder
    file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
    file1.save(file_path1)
    file2.save(file_path2)

    # Process the uploaded images using your Python script
    result1 = process_image(file_path1)
    result2 = process_image(file_path2)

    # Pass the results to the template
    return render_template('result.html', result1=result1, result2=result2)

def process_image(file_path):
    # Add your image processing logic here
    # You can use libraries like OpenCV, Pillow, etc.
    # Example: Resize the image
    from PIL import Image
    img = Image.open(file_path)
    img = img.resize((300, 300))
    processed_path = 'processed_' + file_path
    img.save(processed_path)
    return processed_path

if __name__ == '__main__':
    app.run(debug=True)
