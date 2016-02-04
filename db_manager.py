""" DATABASE MANAGER """

# SESSION

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# engine = create_engine('sqlite:///db.sqlite3', echo=True)
# Session = sessionmaker(bind=engine)


def connect(db):
    engine = create_engine(db, echo=True)
    session = sessionmaker(bind=engine)
    return session()


# MODELS

from sqlalchemy import Column, Integer, String, Boolean, Date
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()


class Entry(Base):
    __tablename__ = 'entry'
    id = Column(Integer, primary_key=True)
    state = Column(String(100))
    date = Column(Date())

# Base.metadata.create_all(engine)
