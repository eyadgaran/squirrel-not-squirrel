'''
Main module for "modeling" endpoints
'''

__author__ = 'Elisha Yadgaran'


from flask import request, render_template, flash, redirect, url_for
import imagehash
from squirrel.database.models import ModelHistory, UserLabel, SquirrelDescription
from simpleml.utils.scoring.load_persistable import PersistableLoader
import base64
import pandas as pd

MODEL = None  #PersistableLoader.load_model('squirrel')  # Debug cloud deployment


def upload(feature):
    if request.method == 'POST' and 'photo' in request.files:
        filename = request.files['photo'].filename
        image_stream = request.files['photo'].stream.read()
        if feature == 'squirrel_not_squirrel':
            history = predict(filename, image_stream)
            negation = '' if history.prediction else 'NOT'
            return render_template('pages/prediction.html', prediction=negation, image=base64.b64encode(image_stream))
        if feature == 'which_squirrel':
            squirrel = get_hash(image_stream)
            return render_template('pages/matching.html', filename=squirrel.filename, description=squirrel.description)
    return render_template('forms/upload.html')


def predict(filename, image_stream):
    x = pd.Series([image_stream])
    prediction_probability = float(MODEL.predict_proba(x)[:, 1])
    prediction = int(round(prediction_probability, 0))

    # DB
    history = ModelHistory.create(
        filename=filename,
        prediction_probability=prediction_probability,
        prediction=prediction
    )

    return history


def get_hash(image_stream):
    hash = imagehash.average_hash(image_stream)
    num_of_pics = len(SquirrelDescription.all())
    pic_id = int(str(hash), 16) % num_of_pics + 1
    return SquirrelDescription.find(pic_id)


def model_feedback():
    user_label = request.form['user_label']
    UserLabel.create(user_label=user_label)
    flash("Thank you for making squirrel-nado smarter!")
    return redirect(url_for('squirrel_not_squirrel'))
