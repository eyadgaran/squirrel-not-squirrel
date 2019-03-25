'''
Module to define the pipeline(s) used
'''

__author__ = 'Elisha Yadgaran'


from simpleml.transformers import Transformer

from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input

import cv2
import math
import numpy as np
import requests
import os


class ImageLoader(Transformer):
    def __init__(self, mode='any', column=None):
        if mode not in ('url', 'file', 'stream', 'any'):
            raise ValueError('Only support: url/file/stream/any')
        self.params = {
            'mode': mode,
            'column': column,
        }

    def load(self, filepath):
        if self.get('mode') == 'any':
            if filepath.startswith('http'):
                raw_img = self.download(filepath)
            elif len(filepath) > 10000:  # Assume binary string
                raw_img = filepath
            else:
                raw_img = self.read(filepath)

        elif self.get('mode') == 'url':
            raw_img = self.download(filepath)

        elif self.get('mode') == 'stream':
            raw_img = filepath

        else:
            raw_img = self.read(filepath)

        # Transform to numpy matrix
        img = cv2.imdecode(np.asarray(bytearray(raw_img), dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def download(self, url):
        req = requests.get(url, stream=True)
        return req.raw.read()

    def read(self, path):
        return open(path, "rb").read()

    def transform(self, X, *args, **kwargs):
        if self.get('column'):
            X.loc[:, self.get('column')] = X[self.get('column')].apply(self.load)
            return X
        else:
            return X.apply(self.load)


class CropImageToSquares(Transformer):
    @staticmethod
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

    def transform(self, X, *args, **kwargs):
        # return np.apply_along_axis(self.image_center_crop, 0, X)
        if self.get('column'):
            X.loc[:, self.get('column')] = X[self.get('column')].apply(self.image_center_crop)
            return X
        else:
            return X.apply(self.image_center_crop)


class ResizeImage(Transformer):
    def __init__(self, column=None, final_dims=(224, 224)):
        self.params = {
            'column': column,
            'final_dims': final_dims,
        }

    def transform(self, X, *args, **kwargs):
        # return np.apply_along_axis(cv2.resize, 0, X, self.final_dims)
        if self.get('column'):
            X.loc[:, self.get('column')] = X[self.get('column')].apply(cv2.resize, args=(self.get('final_dims'),))
            return X
        else:
            return X.apply(cv2.resize, args=(self.get('final_dims'),))


class DataframeToMatrix(Transformer):
    def transform(self, X, *args, **kwargs):
        return np.stack(X.values)


class KerasInceptionV3ImagePreprocessor(Transformer):
    '''Pass through to imagenet preprocessor'''
    def transform(self, X, y=None, **kwargs):
        return preprocess_input(X, **kwargs)


class InceptionV3Encoder(Transformer):
    '''
    Uses Google's inception V3 model as a CNN encoder for image embeddings
    '''
    def __init__(self):
        '''Initializes Architecture and downloads weights'''
        model = InceptionV3(include_top=False)
        self.model = Model(model.inputs, GlobalAveragePooling2D()(model.output))
        self.params = {}

    def transform(self, X, y=None, **kwargs):
        '''
        Encode
        '''
        return self.model.predict(X)


''' Utils '''
# Preprocess for training speed
# Map image files to preencoded ndarrays (dont do direct because mutliple indices map to the same image)
def encode_all_images(df, pipeline, split):
    '''
    Outputs to disk a file named encoded_images.npy
    '''
    # 1) Run through pipeline
    transformed = pipeline.transform(df)

    # 2) Save encoded images
    np.save('encoded_images_' + split, transformed)


def preprocessed_generator(pipeline, split, infinite_loop=False, batch_size=32, shuffle=True, **kwargs):
    X = np.load(os.getenv('ENCODED_IMAGE_PATH', 'encoded_images_') + split + '.npy')
    y = pipeline.get_dataset_split(split)[1].values

    dataset_size = X.shape[0]
    indices = range(dataset_size)

    if dataset_size == 0:  # Return None
        return

    first_run = True
    current_index = 0
    while True:
        if current_index == 0 and shuffle and not first_run:
            np.random.shuffle(indices)

        batch = indices[current_index:min(current_index + batch_size, dataset_size)]
        yield X[batch], y[batch]

        current_index += batch_size

        # Loop so that infinite batches can be generated
        if current_index >= dataset_size:
            if infinite_loop:
                current_index = 0
                first_run = False
            else:
                break
