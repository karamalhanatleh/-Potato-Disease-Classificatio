# Import necessary packages
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO

app = Flask(__name__)
model = tf.keras.models.load_model('potatoes.h5')
class_names = ["Early Blight", "Late Blight", "Healthy"]

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' in request.files:
            image = request.files['image']
            # Convert the FileStorage object to a BytesIO object
            image_stream = BytesIO(image.read())
            # Load and preprocess the image
            image = load_img(image_stream, target_size=(256, 256))  # Adjust the target size according to your model
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0  # Normalize the image data
            result = model.predict(image)
            class_index = np.argmax(result)  # Get the index of the class with the highest probability
            class_name = class_names[class_index]  # Get the class name from the list
            return f'This image is classified as: {class_name}'
        else:
            return 'Image not found.'
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return 'An error occurred while processing the image.'

if __name__ == '__main__':
    app.run(debug=True)
