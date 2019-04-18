'''
Module to define the model(s) used
'''

__author__ = 'Elisha Yadgaran'


from simpleml.models import KerasSequentialClassifier
from simpleml.utils.errors import ModelError
from simpleml import TRAIN_SPLIT, VALIDATION_SPLIT

from keras.layers import Dense, Dropout

import logging
LOGGER = logging.getLogger(__name__)


class GeneratorModel(KerasSequentialClassifier):
    '''
    Extends Base Keras Model to support data generators for fitting

    ONLY use with supporting pipeline!
    '''
    def fit(self, train_generator=None, validation_generator=None, **kwargs):
        '''
        Pass through method to external model after running through pipeline

        Optionally overwrite normal method to pass in the generator directly. Used
        to speedup training by caching the transformed input before training the
        model - avoids downloading, reading, encoding images in every batch
        '''
        if self.pipeline is None:
            raise ModelError('Must set pipeline before fitting')

        if self.state['fitted']:
            LOGGER.warning('Cannot refit model, skipping operation')
            return self
        if train_generator is None:
            # Explicitly fit only on train split
            train_generator = self.pipeline.transform(X=None, dataset_split=TRAIN_SPLIT, return_y=True, infinite_loop=True, **self.get_params())
            validation_generator = self.pipeline.transform(X=None, dataset_split=VALIDATION_SPLIT, return_y=True, infinite_loop=True, **self.get_params())

        self._fit(train_generator, validation_generator)

        # Mark the state so it doesnt get refit and can now be saved
        self.state['fitted'] = True

        return self

    def _fit(self, train_generator, validation_generator=None):
        '''
        Keras fit parameters (epochs, callbacks...) are stored as self.params so
        retrieve them automatically
        '''
        # Generator doesnt take arbitrary params so pop the extra ones
        extra_params = ['batch_size']
        params = {k:v for k, v in self.get_params().items() if k not in extra_params}
        self.external_model.fit_generator(
            generator=train_generator, validation_data=validation_generator, **params)


class RetrainedTopModel(GeneratorModel):
    '''
    Retrained top layer of some transfer learned model
    '''
    def build_network(self, model, **kwargs):
        '''
        training network

        Input:
            X = [image embeddings]
            y = [labels]

        Output:
            y = [predictions]
        '''
        IMG_EMBED_SIZE = 2048  # InceptionV3 output

        # Save config values for later
        new_configs =  {
                'IMG_EMBED_SIZE': IMG_EMBED_SIZE,
        }
        self.config.update(new_configs)

        ###############
        # Image Input #
        ###############
        # [batch_size, IMG_EMBED_SIZE] of CNN image features
        # model.add(Input(shape=(IMG_EMBED_SIZE,), dtype='float32', name='image_input'))
        model.add(Dense(1024, activation='relu', name='top_layer1',
                        input_shape=(IMG_EMBED_SIZE,), dtype='float32'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu', name='top_layer2'))
        model.add(Dense(512, activation='relu', name='top_layer3'))
        model.add(Dense(2, activation='softmax', name='prediction'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(model.summary())
        # from keras.utils.vis_utils import plot_model
        # plot_model(model, to_file='model.png', show_shapes=True)

        return model
