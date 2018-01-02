from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric, DateTime, func
from sqlalchemy_mixins import AllFeaturesMixin
from src.app import app

engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
db_session = scoped_session(sessionmaker(autocommit=True,
                                         autoflush=False,
                                         bind=engine))
Base = declarative_base()
# Base.query = db_session.query_property()


class BaseModel(Base, AllFeaturesMixin):
    __abstract__ = True
    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), server_onupdate=func.now())


class Feedback(BaseModel):
    __tablename__ = 'requests'

    id = Column(Integer, primary_key=True)
    feedback = Column(String(1200), nullable=False)


class ModelHistory(BaseModel):
    __tablename__ = 'model_history'

    id = Column(Integer, primary_key=True)
    filename = Column(String(), unique=True)
    prediction_probability = Column(Numeric)
    prediction = Column(Integer)
    user_label = Column(Integer)
    label = Column(Integer)


# Create tables.
# Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
BaseModel.set_session(db_session)
