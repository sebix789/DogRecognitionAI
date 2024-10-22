from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import os
import random
from dotenv import load_dotenv
from model.model import load_model

load_dotenv()
app = Flask(__name__)
CORS(app)

model = load_model()
calibration_data = {}


def prepare_image(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img


@app.route('/calibrate', methods=['POST'])
def calibrate():
    pet_name = request.form.get('pet_name')
    calibration_images = request.files.getlist('calibration_images')
    
    if pet_name not in calibration_data:
        calibration_data[pet_name] = []
    
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    calibration_path = []
    for img in calibration_images:
        img_path = os.path.join('temp', img.filename)
        img.save(img_path)
        calibration_path.append(img_path)

    calibration_data[pet_name].extend(calibration_path)

    return jsonify({'message': 'Model Calibrated for pet: {}'.format(pet_name)})

@app.route('/predict', methods=['POST'])
def predict():
    pet_name = request.form.get('pet_name')
    prediction_image = request.files['prediction_image']
    
    if pet_name not in calibration_data or not calibration_data[pet_name]:
        return jsonify({'error': f'No calibration data found for pet: {pet_name}'}), 400

    if not os.path.exists('temp'):
        os.makedirs('temp')

    prediction_path = os.path.join('temp', prediction_image.filename)
    prediction_image.save(prediction_path)
    prepared_prediction_image = prepare_image(prediction_path)

    calibration_images = calibration_data[pet_name]
    random_calibration_image_path = random.choice(calibration_images)
    prepared_calibration_image = prepare_image(random_calibration_image_path)

    prediction_embedding = model.predict([prepared_prediction_image, prepared_calibration_image])
    distance = np.linalg.norm(prediction_embedding)

    os.remove(prediction_path)
    
    print(f'Distance: {distance}')

    return jsonify({'distance': float(distance)})
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)