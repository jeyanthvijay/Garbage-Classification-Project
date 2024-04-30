from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import os
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Define label mapping
label_mapping = {
    0: 'cardboard',
    1: 'compost',
    2: 'glass',
    3: 'metal',
    4: 'paper',
    5: 'plastic',
    6: 'trash'
}

# Load the trained model
model = load_model("Garbage1.h5")

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Home page route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']

    # Check if the file is empty
    if file.filename == '':
        return 'No selected file', 400

    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return 'File extension not allowed', 400

    # Save the file to the upload folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Load the image and preprocess it
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make a prediction using the pre-trained model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        result = label_mapping[predicted_class]

        # Render the result page with the prediction and image path
        return render_template('results.html', result=result, image=file_path)

    except Exception as e:
        return f"Error: {str(e)}", 500

    finally:
        # Remove the uploaded file after processing
        os.remove(file_path)


if __name__ == '__main__':
    app.run(debug=False)
