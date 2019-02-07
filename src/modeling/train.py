'''
Module for model training
'''

__author__ = 'Elisha Yadgaran'


# import plaidml.keras
# plaidml.keras.install_backend()
# import logging
# logging.getLogger("plaidml").setLevel(logging.CRITICAL)

from simpleml.utils.training.create_persistable import \
    DatasetCreator, PipelineCreator, ModelCreator, MetricCreator
from simpleml import TRAIN_SPLIT, TEST_SPLIT
from itertools import product
from simpleml.utils.scoring.load_persistable import PersistableLoader
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from src.database.initialization import SimpleMLDatabase
from src.modeling.dataset import SquirrelDataset


def train():
    # Initialize session
    SimpleMLDatabase().initialize()

    # Build the Dataset
    dataset_kwargs = {'project': 'squirrel', 'name': 'squirrel', 'registered_name': 'SquirrelDataset',
                          'label_columns': ['y'], 'strict': False, 'save_method': 'disk_hdf5'}
    pipeline_kwargs = {'project': 'squirrel', 'name': 'squirrel', 'registered_name': 'BaseRandomSplitProductionPipeline',
                       'train_size': 0.8, 'validation_size': 0.0, 'test_size': 0.2, 'shuffle': True, 'random_state': 38, 'strict': False}

    # early = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=1, verbose=1, mode='auto')
    model_kwargs = {'project': 'squirrel', 'name': 'squirrel', 'registered_name': 'VGGExtendedKerasModel',
                    'params': {'batch_size': 32, 'validation_split': 0.1, 'epochs': 30, 'callbacks': []}}

    dataset = DatasetCreator.retrieve_or_create(**dataset_kwargs)
    pipeline = PipelineCreator.retrieve_or_create(dataset=dataset, **pipeline_kwargs)
    model = ModelCreator.retrieve_or_create(pipeline=pipeline, **model_kwargs)

    # Evaluate Metrics
    metrics_to_score = ['RocAucMetric', 'F1ScoreMetric', 'AccuracyMetric', 'TprMetric', 'FprMetric']
    dataset_splits = [TRAIN_SPLIT, TEST_SPLIT]
    for cls, dataset_split in product(metrics_to_score, dataset_splits):
        metric_kwargs = {'registered_name': cls, 'dataset_split': dataset_split}
        metric = MetricCreator.retrieve_or_create(model=model, **metric_kwargs)
        print metric.name, metric.values


if __name__ == '__main__':
    train()
