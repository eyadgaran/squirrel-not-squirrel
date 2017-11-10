import Flask
from Flask import request, jsonify
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np


app = Flask(__name__)

model = load_model('retrained_model')


@app.route('./predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)
    # TODO: normalize the prediction here
    return jsonify({'prediction': prediction}), 200
