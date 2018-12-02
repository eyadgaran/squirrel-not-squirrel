'''
Module for database initialization
'''

__author__ = 'Elisha Yadgaran'


from simpleml.utils.initialization import Database
from models import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker


class SimpleMLDatabase(object):
    def __init__(self):
        self.database_params = {
            'database': 'SQUIRREL-ML',
            'user': 'squirrel',
            'password': None,
            'jdbc': 'postgresql',
            'host': 'localhost',
            'port': 5432
        }

    def initialize(self, create_objects=False):
        db = Database(**self.database_params)
        db.initialize(create_database=create_objects)


class AppDatabase(object):
    def __init__(self):
        self.database_params = {
            'database': 'SQUIRREL',
            'user': 'squirrel',
            'password': None,
            'jdbc': 'postgresql',
            'host': 'localhost',
            'port': 5432
        }

    def initialize(self):
        url = '{jdbc}://{user}:{password}@{host}:{port}/{database}'.format(
            **self.database_params)
        engine = create_engine(url)
        db_session = scoped_session(sessionmaker(autocommit=True,
                                                 autoflush=False,
                                                 bind=engine))
        # Create tables.
        BaseModel.metadata.create_all(bind=engine)
        BaseModel.set_session(db_session)
