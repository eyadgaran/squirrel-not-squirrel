from flask import request, render_template, flash, redirect, url_for
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import imagehash
import numpy as np
from src.app import app
import os
from src.database.models import ModelHistory, UserLabel, SquirrelDescription
from simpleml.utils.scoring.load_persistable import PersistableLoader
import cv2
import math
import base64


MODEL = PersistableLoader.load_model('squirrel')

@app.route('/upload/<feature>', methods=['GET', 'POST'])
def upload(feature):
    if request.method == 'POST' and 'photo' in request.files:
        filename = request.files['photo'].filename
        raw_image = request.files['photo'].stream.read()
        if feature == 'squirrel_not_squirrel':
            history = predict(filename, raw_image)
            negation = '' if history.prediction else 'NOT'
            return render_template('pages/prediction.html', prediction=negation, image=base64.b64encode(raw_image))
        if feature == 'which_squirrel':
            squirrel = get_hash(filename)
            return render_template('pages/matching.html', filename=squirrel.filename, description=squirrel.description)
    return render_template('forms/upload.html')


def read_image(raw_img):
    # Transform to numpy matrix
    img = cv2.imdecode(np.asarray(bytearray(raw_img), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_center_crop(img):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.
    Returns [min(h, w), min(h, w), 3] output with same width and height.
    """

    h = min(img.shape[0], img.shape[1])
    w = min(img.shape[0], img.shape[1])

    h_padding = (img.shape[0] - h) / 2.
    w_padding = (img.shape[1] - w) / 2.

    cropped_img = img[int(math.floor(h_padding)): int(img.shape[0] - math.ceil(h_padding)),
                      int(math.floor(w_padding)): int(img.shape[1] - math.ceil(w_padding)),
                      :]

    return cropped_img

def resize_image(image):
    return cv2.resize(image, (224, 224))

def predict(filename, raw_img):
    img = read_image(raw_img)
    img = image_center_crop(img)
    x = resize_image(img)

    x = preprocess_input(x, mode='tf')
    x = np.stack([x])
    prediction_probability = float(MODEL.predict(x))
    prediction = int(round(prediction_probability, 0))

    # DB
    history = ModelHistory.create(
        filename=filename,
        prediction_probability=prediction_probability,
        prediction=prediction
    )

    return history


def get_hash(filename):
    file_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
    hash = imagehash.average_hash(Image.open(file_path))
    num_of_pics = len(SquirrelDescription.all())
    pic_id = int(str(hash), 16) % num_of_pics + 1
    return SquirrelDescription.find(pic_id)


@app.route('/record_model_feedback', methods=['POST'])
def model_feedback():
    user_label = request.form['user_label']
    UserLabel.create(user_label=user_label)
    flash("Thank you for making squirrel-nado smarter!")
    return redirect(url_for('squirrel_not_squirrel'))
