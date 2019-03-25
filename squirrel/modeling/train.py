'''
Module for model training
'''

__author__ = 'Elisha Yadgaran'


# import plaidml.keras
# plaidml.keras.install_backend()
# import logging
# logging.getLogger("plaidml").setLevel(logging.CRITICAL)

from simpleml.utils import \
    DatasetCreator, PipelineCreator, ModelCreator, MetricCreator
from simpleml import TRAIN_SPLIT, TEST_SPLIT
from itertools import product
from simpleml.utils import PersistableLoader
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from squirrel.database.initialization import SimpleMLDatabase
from .dataset import *
from .pipeline import *
from .model import *


def train():
    # Initialize session
    SimpleMLDatabase().initialize()

    # Build the Dataset
    dataset_kwargs = {'project': 'squirrel', 'name': 'squirrel', 'registered_name': 'SquirrelDataset',
                          'label_columns': ['label'], 'strict': False, 'save_method': 'cloud_pickled'}
    pipeline_kwargs = {
        'project': 'squirrel', 'name': 'squirrel', 'registered_name': 'RandomSplitPipeline',
        'fitted': True, 'train_size': 0.8, 'validation_size': 0.0, 'test_size': 0.2,
        'shuffle': True, 'random_state': 38, 'strict': False, 'save_method': 'cloud_pickled',
        'transformers': [
             ('load_images', ImageLoader()),
             ('crop', CropImageToSquares()),
             ('resize', ResizeImage(final_dims=(224, 224))),
             ('df_to_matrix', DataframeToMatrix()),
             ('preprocess_tuple', KerasInceptionV3ImagePreprocessor()),
             ('encode', InceptionV3Encoder()),
        ]
    }

    early = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=3, verbose=1, mode='auto')
    model_kwargs = {
        'project': 'squirrel', 'name': 'squirrel', 'registered_name': 'RetrainedTopModel',
        'save_method': 'cloud_keras_hdf5',
        'params': {'batch_size': 32, 'callbacks': [early],
                   'epochs': 500, 'steps_per_epoch': 100, 'validation_steps': 50,
                   'use_multiprocessing': False, 'workers': 0}
    }

    dataset = DatasetCreator.retrieve_or_create(**dataset_kwargs)
    pipeline = PipelineCreator.retrieve_or_create(dataset=dataset, **pipeline_kwargs)
    # Preprocess for training speed
    # encode_all_images(pipeline.get_dataset_split(TRAIN_SPLIT)[0], pipeline, TRAIN_SPLIT)
    # encode_all_images(pipeline.get_dataset_split(TEST_SPLIT)[0], pipeline, TEST_SPLIT)
    # model = ModelCreator.retrieve_or_create(pipeline=pipeline, **model_kwargs)

    # Use preprocessed data
    model = RetrainedTopModel(**model_kwargs)
    model.add_pipeline(pipeline)
    train_generator = preprocessed_generator(pipeline, split=TRAIN_SPLIT, return_y=True, infinite_loop=True, **model.get_params())
    validation_generator = preprocessed_generator(pipeline, split=TEST_SPLIT, return_y=True, infinite_loop=True, **model.get_params())
    model.fit(train_generator, validation_generator)
    model.params.pop('callbacks', [])
    model.save()

    # Evaluate Metrics
    metrics_to_score = ['RocAucMetric', 'F1ScoreMetric', 'AccuracyMetric', 'TprMetric', 'FprMetric']
    dataset_splits = [TRAIN_SPLIT, TEST_SPLIT]
    for cls, dataset_split in product(metrics_to_score, dataset_splits):
        metric_kwargs = {'registered_name': cls, 'dataset_split': dataset_split}
        metric = MetricCreator.retrieve_or_create(model=model, **metric_kwargs)
        print(metric.name, metric.values)


if __name__ == '__main__':
    train()
