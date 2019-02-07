'''
Module to define the dataset(s) used for training and validation
'''

__author__ = 'Elisha Yadgaran'


from simpleml.datasets import NumpyDataset

import os
import numpy as np
import requests
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from tqdm import tqdm


current_directory = os.path.dirname(os.path.realpath(__file__))
NEGATIVE_IMAGE_DIRECTORY = os.path.abspath(os.path.join(current_directory, '../../data/negative/'))
POSITIVE_IMAGE_DIRECTORY = os.path.abspath(os.path.join(current_directory, '../../data/positive/'))
IMAGENET_LINK_DIRECTORY = os.path.abspath(os.path.join(current_directory, '../../data/imagenet_links/'))
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
IMAGENET_POSITIVE_LABEL = 'squirrel'


class ImageLoadingDataset(NumpyDataset):
    def download_images(self):
        # Check if already downloaded before doing anything
        already_downloaded = self.state.get('links_downloaded', False)
        if already_downloaded:
            return

        # Load Txt links
        link_dictionary = {}
        for filename in os.listdir(IMAGENET_LINK_DIRECTORY):
            with open(os.path.join(IMAGENET_LINK_DIRECTORY, filename)) as f:
                link_dictionary[filename[:-4]] = [x.strip() for x in f.readlines()]

        # Split into "positive" and "negative" lists
        positive_links = link_dictionary[IMAGENET_POSITIVE_LABEL]
        negative_links = []

        # There are duplicates, unfortunately, so have to dedupe
        for class_label, link_list in link_dictionary.iteritems():
            if class_label == IMAGENET_POSITIVE_LABEL:
                continue
            negative_links.extend([item for item in link_list if item not in positive_links])

        for link_list, directory in zip([positive_links, negative_links], [POSITIVE_IMAGE_DIRECTORY, NEGATIVE_IMAGE_DIRECTORY]):
            for link in link_list:
                filename = link.rsplit('/', 1)[-1]
                try:
                    response = requests.get(link)
                    if response.status_code == 200:
                        with open(os.path.join(directory, filename), 'wb') as f:
                            f.write(response.content)
                except Exception as e: #requests.exceptions.ConnectionError as e:
                    print e

        # Make note that links were downloaded so it wont do it again
        self.state['links_downloaded'] = True

    def load_images(self, directory_path, label):
        image_list = []
        for filename in tqdm(os.listdir(directory_path)):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath):
                try:
                    img = image.load_img(filepath, target_size=(224, 224))
                    x = image.img_to_array(img)
                    # Need to store arrays as a list because numpy doesnt have a hash function
                    image_list.append(preprocess_input(x, mode='tf'))
                except IOError as e:
                    print e

        image_tuple = (np.stack(image_list), np.repeat(label, len(image_list)))

        return image_tuple


class SquirrelDataset(ImageLoadingDataset):
    def build_dataframe(self):
        # self.download_images()
        negative_matrix, negative_label = self.load_images(NEGATIVE_IMAGE_DIRECTORY, NEGATIVE_LABEL)
        positive_matrix, positive_label = self.load_images(POSITIVE_IMAGE_DIRECTORY, POSITIVE_LABEL)

        self._external_file = {
            'X': np.concatenate((negative_matrix, positive_matrix)),
            'y': np.concatenate((negative_label, positive_label))
        }
