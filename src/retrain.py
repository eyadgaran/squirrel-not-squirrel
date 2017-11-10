from vgg16 import VGG16
from keras import Model
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def load_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    return preprocess_input(x, mode='tf')


def load_dataset():
    X = []
    y = []
    for filename in os.listdir('../data/'):
        if filename.startswith('ne'):
            X.append(load_img(os.path.join('../data', filename)))
            y.append(0)
        if filename.startswith('po'):
            X.append(load_img(os.path.join('../data', filename)))
            y.append(1)
    X = np.array(X)
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, shuffle=True, random_state=38)


X_train, X_test, y_train, y_test = load_dataset()


base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dense(1, activation='softmax'))
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# for layer in model.layers[:15]:
#     layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=16, epochs=10)
model.save('retrained_model')
y_pred = model.predict(X_test)
print(f1_score(y_test, y_pred))
