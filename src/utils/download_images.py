import os
import requests

current_directory = os.path.dirname(os.path.realpath(__name__))
data_directory = os.path.join(current_directory, '../../data')
imagenet_files = os.path.join(data_directory, 'imagenet_links')
target_class = 'squirrel'


def load_image_links():
    link_dictionary = {}
    for filename in os.listdir(imagenet_files):
        with open(os.path.join(imagenet_files, filename)) as f:
            link_dictionary[filename[:-4]] = [x.strip() for x in f.readlines()]
    return link_dictionary


def label_links(links):
    positive_links = links[target_class]
    negative_links = []

    for class_label, link_list in links.iteritems():
        if class_label == target_class:
            continue
        negative_links.extend([item for item in link_list if item not in positive_links])

    return positive_links, negative_links


def download_links(links, label):
    download_directory = os.path.join(data_directory, label)
    os.mkdir(download_directory)

    for url in links:
        filename = url.rsplit('/', 1)[-1]
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(download_directory, filename), 'wb') as f:
                    f.write(response.content)
        except Exception as e: #requests.exceptions.ConnectionError as e:
            pass


if __name__ == '__main__':
    links = load_image_links()
    positive_links, negative_links = label_links(links)
    download_links(positive_links, 'positive')
    download_links(negative_links, 'negative')
