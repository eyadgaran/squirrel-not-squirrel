from vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

current_directory = os.path.dirname(os.path.realpath(__name__))
data_directory = os.path.abspath(os.path.join(current_directory, '../../data'))


def load_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    return preprocess_input(x, mode='tf')


def load_dataset():
    X = []
    y = []
    for filename in os.listdir(data_directory):
        if not os.path.isfile(os.path.join(data_directory, filename)):
            continue
        if filename.startswith('ne'):
            X.append(load_img(os.path.join(data_directory, filename)))
            y.append(0)
        if filename.startswith('po'):
            X.append(load_img(os.path.join(data_directory, filename)))
            y.append(1)

    # Imagenet files
    for path, label in zip(['positive', 'negative'], [1, 0]):
        filepath = os.path.join(data_directory, path)
        for filename in os.listdir(filepath):
            if filename[-4:] != '.jpg':
                continue
            try:
                X.append(load_img(os.path.join(filepath, filename)))
                y.append(label)
            except IOError as e:
                print e

    X = np.array(X)
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, shuffle=True, random_state=38)


X_train, X_test, y_train, y_test = load_dataset()


base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=50)
model.save('retrained_model')
y_pred = model.predict(X_test)
print f1_score(y_test, np.squeeze(y_pred).round())
