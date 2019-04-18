'''
Module for database tables
'''

__author__ = 'Elisha Yadgaran'


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric, DateTime, func, BigInteger
from sqlalchemy_mixins import AllFeaturesMixin


Base = declarative_base()


class BaseModel(Base, AllFeaturesMixin):
    __abstract__ = True
    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), nullable=True, server_onupdate=func.now())
    id = Column(BigInteger, primary_key=True)


class Feedback(BaseModel):
    __tablename__ = 'requests'

    feedback = Column(String(1200), nullable=False)


class ModelHistory(BaseModel):
    __tablename__ = 'model_history'

    filename = Column(String())
    prediction_probability = Column(Numeric)
    prediction = Column(Integer)
    user_label = Column(Integer)
    label = Column(Integer)


class UserLabel(BaseModel):
    __tablename__ = 'user_labels'

    user_label = Column(String(12), nullable=False)


class SquirrelDescription(BaseModel):
    __tablename__ = "squirrel_descriptions"

    filename = Column(String(), unique=True)
    description = Column(String())
