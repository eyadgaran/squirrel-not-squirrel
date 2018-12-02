'''
Module to define the model(s) used
'''

__author__ = 'Elisha Yadgaran'


from simpleml.models.classifiers.keras.sequential import KerasSequentialClassifier

# from keras.applications import VGG16
from vgg16 import VGG16
from keras.optimizers import SGD
from keras.layers import Dense, Flatten, Dropout
from keras.models import clone_model


class VGGExtendedKerasModel(KerasSequentialClassifier):
    @staticmethod
    def clone_network(model, drop_layers=None):
        # Tensorflow graphs are append only so popping layers causes errors
        # when get_config is called for saving/loading
        model_copy = clone_model(model)
        model_copy.set_weights(model.get_weights())

        return model_copy

    def build_network(self, model, **kwargs):
        transfer_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        temp_model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
        base_model = self.clone_network(transfer_model)

        # Drop classifier layer
        for layer in base_model.layers:
            layer.trainable = False

        # Hack to copy over FC layers without breaking graph
        model.add(base_model)
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dense(4096, activation='relu', name='fc2'))

        # Copy over weights and freeze
        for layer in ['flatten', 'fc1', 'fc2']:
            model.get_layer(layer).set_weights(temp_model.get_layer(layer).get_weights())
            model.get_layer(layer).trainable = False

        # Add our own layers and classifier
        model.add(Dropout(0.6))
        model.add(Dense(2, activation='softmax'))

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        print model.summary()

        return model
