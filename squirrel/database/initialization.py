'''
Module for database initialization
'''

__author__ = 'Elisha Yadgaran'


from simpleml.utils import Database
from .models import BaseModel


class SimpleMLDatabase(Database):
    def __init__(self):
        super(SimpleMLDatabase, self).__init__(configuration_section='simpleml-squirrel')


class AppDatabase(Database):
    def __init__(self):
        super(AppDatabase, self).__init__(configuration_section='app-squirrel')

    def initialize(self, **kwargs):
        self._initialize(BaseModel, **kwargs)
