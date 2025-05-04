from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'runs/detect/train3/weights/best.pt'  # Update this path

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load the model
model = YOLO(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser submits empty file
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Run inference
            results = model.predict(
                source=upload_path,
                conf=0.5,  # confidence threshold
                save=True,
                project=app.config['RESULT_FOLDER'],
                name='predictions',
                exist_ok=True
            )
            
            # Get the path to the result image
            result_filename = f"predictions/{filename}"
            return render_template('result.html', 
                                original=url_for('static', filename=f'uploads/{filename}'),
                                result=url_for('static', filename=f'results/{result_filename}'))
    
    return render_template('upload.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='results/predictions/' + filename), code=301)

if __name__ == '__main__':
    # Create folders if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)