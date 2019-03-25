'''
Module to define the dataset(s) used for training and validation
'''

__author__ = 'Elisha Yadgaran'


from simpleml.datasets import PandasDataset

import os
import numpy as np
import pandas as pd
import requests
import cv2
from tqdm import tqdm


current_directory = os.path.dirname(os.path.realpath(__file__))
NEGATIVE_IMAGE_DIRECTORY = os.path.abspath(os.path.join(current_directory, '../../data/negative/'))
POSITIVE_IMAGE_DIRECTORY = os.path.abspath(os.path.join(current_directory, '../../data/positive/'))
IMAGENET_LINK_DIRECTORY = os.path.abspath(os.path.join(current_directory, '../../data/imagenet_links/'))
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
IMAGENET_POSITIVE_LABEL = 'squirrel'


class ImageLoadingDataset(PandasDataset):
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
                    print(e)

        # Make note that links were downloaded so it wont do it again
        self.state['links_downloaded'] = True

    def load_images(self, directory_path, label):
        file_list = []
        for filename in tqdm(os.listdir(directory_path)):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath):
                try:  # Attempt to load files because many are corrupted or blank
                    img = cv2.imdecode(np.asarray(bytearray(open(filepath, "rb").read()), dtype=np.uint8), 1)
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    file_list.append(filepath.decode('UTF-8'))
                except (IOError, cv2.error) as e:
                    print(e)

        return pd.DataFrame(list(zip(file_list, [label] * len(file_list))),
                            columns=['image', 'label'])


class SquirrelDataset(ImageLoadingDataset):
    def build_dataframe(self):
        # self.download_images()
        negative_df = self.load_images(NEGATIVE_IMAGE_DIRECTORY, NEGATIVE_LABEL)
        positive_df = self.load_images(POSITIVE_IMAGE_DIRECTORY, POSITIVE_LABEL)

        self._external_file = pd.concat([negative_df, positive_df], axis=0)
