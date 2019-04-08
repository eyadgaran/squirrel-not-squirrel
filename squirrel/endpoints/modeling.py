'''
Main module for "modeling" endpoints
'''

__author__ = 'Elisha Yadgaran'


from quart import request, render_template, flash, redirect, url_for
import imagehash
from squirrel.database.models import ModelHistory, UserLabel, SquirrelDescription
from simpleml.utils.scoring.load_persistable import PersistableLoader
import base64
import pandas as pd
import asyncio
import threading
import tensorflow as tf


class ModelWrapper(object):
    '''
    Lot of hackery to get the model to load in parallel when the service
    starts up

    Had trouble getting asyncio to actually execute in parallel so hacked the following:
    1) Load in thread
    2) Create new event loop for thread
    3) Save graph from thread to use in main thread at predict time
    '''
    def __init__(self):
        self._model = None
        self.concurrent_load_model()

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    def predict_proba(self, *args):
        with self.graph.as_default():
            return self.model.predict_proba(*args)

    def load_model(self):
        self._model = PersistableLoader.load_model('squirrel')
        self._model.load(load_externals=True)
        self.graph = tf.get_default_graph()

    def async_load_model(self):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        self.load_model()

    def concurrent_load_model(self):
        t = threading.Thread(target=self.async_load_model)
        t.daemon = True
        t.start()

MODEL = ModelWrapper()


async def upload(feature):
    files = await request.files
    if request.method == 'POST' and 'photo' in files:
        filename = files['photo'].filename
        image_stream = files['photo'].stream.read()
        if feature == 'squirrel_not_squirrel':
            history = await predict(filename, image_stream)
            negation = '' if history.prediction else 'NOT'
            # .decode is necessary on python 3 for bytes to str conversion
            return await render_template('pages/prediction.html', prediction=negation, image=base64.b64encode(image_stream).decode())
        if feature == 'which_squirrel':
            squirrel = get_hash(image_stream)
            return await render_template('pages/matching.html', filename=squirrel.filename, description=squirrel.description)
    return await render_template('forms/upload.html')


async def predict(filename, image_stream):
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


async def model_feedback():
    form = await request.form
    user_label = form['user_label']
    UserLabel.create(user_label=user_label)
    await flash("Thank you for making squirrel-nado smarter!")
    return redirect(url_for('squirrel_not_squirrel'))
