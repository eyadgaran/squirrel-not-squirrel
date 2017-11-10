from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import numpy as np

model = VGG16(include_top=True, weights='imagenet')

img_path = 's1.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
is_squirrel = False
for _, label, prob in decode_predictions(preds)[0]:
    if 'squirrel' in label:
        is_squirrel = True

outcome = "SQUIRREL" if is_squirrel else "NOT A SQUIRREL"
print outcome
