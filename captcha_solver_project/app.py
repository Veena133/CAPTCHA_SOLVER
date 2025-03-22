import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
# to run: python app.py
# Define the Flask app
app = Flask(__name__)

# Assuming CTCLayer is already defined as in the provided code
from tensorflow.keras import layers

class CTCLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(CTCLayer, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=None, logit_length=None))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

# Load the trained model (ensure the custom CTCLayer is included)
model = load_model('model/my_model.h5', custom_objects={'CTCLayer': CTCLayer})

# Function to prepare a new image for prediction
def prepare_image(img_path, img_width=200, img_height=50):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    
    input_length = np.ones((img.shape[0], 1), dtype=np.int32) * img.shape[1]
    
    return img, input_length

# Function to decode the prediction output
def decode_prediction(prediction, num_to_char):
    input_length = np.ones(prediction.shape[0]) * prediction.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(prediction, input_length, greedy=True)
    
    decoded_text = []
    for seq in decoded[0]:
        text = ''.join([num_to_char(char.numpy()).numpy().decode("utf-8") for char in seq if char != -1])
        decoded_text.append(text)
    
    return decoded_text[0]

# Utility function for mapping numbers to characters
def num_to_char(num_tensor):
    char_map = "12345678bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char_map_tensor = tf.constant(list(char_map), dtype=tf.string)

    num_tensor = tf.clip_by_value(num_tensor, 0, len(char_map) - 1)
    
    return tf.gather(char_map_tensor, tf.cast(num_tensor, dtype=tf.int32))
# Routes
@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/main')
def main():
    return render_template('index.html')
# Endpoint for file upload and prediction
@app.route('/predict', methods=['POST'])
def predict_captcha():
    if 'captcha_image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['captcha_image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded image temporarily
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    img_path = os.path.join(upload_folder, file.filename)
    file.save(img_path)
    
    # Prepare the image and make a prediction
    img, input_length = prepare_image(img_path)
    
    prediction_model = tf.keras.models.Model(
        model.input[0], model.get_layer(name="dense2").output
    )
    prediction = prediction_model.predict(img)
    decoded_text = decode_prediction(prediction, num_to_char)
    
    # Delete the uploaded image after prediction
    os.remove(img_path)
    
    # Return the predicted text as a response
    return decoded_text, 200

# @app.route('/welcome')
# def welcome():
#     return render_template('welcome.html')
# # Frontend form to upload an image

# @app.route('/')
# def main():
#     return render_template('index.html')
# @app.route('/')
# def home():
#     return welcome()
if __name__ == "__main__":
    app.run(debug=True)
