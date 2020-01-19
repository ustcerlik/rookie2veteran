import datetime
import logging

from sqlalchemy import Column, Integer, String, create_engine, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
logger = logging.getLogger()


class DetData(Base):
    __tablename__ = "detdata"

    id = Column(Integer, primary_key=True)
    scene_id = Column(Integer)
    channel_id = Column(Integer)  #
    label_id = Column(Integer)  #
    label_type = Column(String(20))  # bbox mask keypoint
    camera = Column(String(11))  # fid did fisheye
    hdfs_path = Column(String(200))  # channel_data.tar for det,
    hdfs_anno_path = Column(String(200), unique=True)  # channel_date_label.json
    collect_data = Column(DateTime)  # data time 20190925
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)
    img_num = Column(Integer)

    def to_dict(self):
        dict_info = {"scene_id": self.scene_id,
                     "channel_id": self.channel_id,
                     "label_id": self.label_id,
                     "label_type": self.label_type,
                     "camera": self.camera,
                     "hdfs_path": self.hdfs_path,
                     "hdfs_anno_path": self.hdfs_anno_path,
                     "img_num": self.img_num,
                     "table": self.__tablename__}
        if self.id:
            dict_info["id"] = self.id

        return dict_info

    @classmethod
    def get_column(cls, item):
        return object.__getattribute__(cls, item)


class DetValData(Base):
    __tablename__ = "detvaldata"

    id = Column(Integer, primary_key=True)  # suggested query key
    labels = Column(String(200))
    label_type = Column(String(20))  # bbox mask keypoint
    camera = Column(String(11))  # fid did fisheye general
    hdfs_path = Column(String(200))  # anything.tar for det
    hdfs_anno_path = Column(String(200), unique=True)  # anything.json for det
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)
    img_num = Column(Integer)
    unique_key = Column(String(200), unique=True)  # easily find a val data through this description,

    def to_dict(self):
        dict_info = {"unique_key": self.unique_key,
                     "label_type": self.label_type,
                     "labels": self.labels,
                     "camera": self.camera,
                     "hdfs_path": self.hdfs_path,
                     "hdfs_anno_path": self.hdfs_anno_path,
                     "img_num": self.img_num,
                     "table": self.__tablename__}
        if self.id:
            dict_info["id"] = self.id

        return dict_info

    @classmethod
    def get_column(cls, item):
        return object.__getattribute__(cls, item)


class FaceData(Base):
    __tablename__ = "data"

    id = Column(Integer, primary_key=True)
    scene_id = Column(Integer)
    camera = Column(String(11))  # fid did fisheye
    hdfs_path = Column(String(200))
    hdfs_anno_path = Column(String(200), unique=True)
    collect_data = Column(DateTime)  # data time 20190925
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)
    img_num = Column(Integer)
    hdfs_img_raw = Column(String(200))
    pid_num = Column(Integer)
    anno_filter_ceph = Column(String(200))
    benchmark_used = Column(String(200))


class Customer(Base):
    __tablename__ = "customer"
    id = Column(Integer, primary_key=True)
    name = Column(String(20), unique=True)  # wanda anta nanling ctf
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)
    customer_type = Column(String(30))  # TODO mall store general ??? FACE????

    @classmethod
    def get_column(cls, item):
        return object.__getattribute__(cls, item)


class Scene(Base):
    __tablename__ = 'scene'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    customer_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)

    @classmethod
    def get_column(cls, item):
        return object.__getattribute__(cls, item)


class DetLabel(Base):
    __tablename__ = "detlabel"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)  # body head realface headwithoutgoodface headwithgoodface
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)

    @classmethod
    def get_column(cls, item):
        return object.__getattribute__(cls, item)


class Channel(Base):
    __tablename__ = "channel"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)  # TODO ch01001 ? can this be unique ?
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)

    @classmethod
    def get_column(cls, item):
        return object.__getattribute__(cls, item)


class DBClient(object):
    def __init__(self, group="detection"):
        super(DBClient, self).__init__()
        self.group = group
        self.engine = self.create_engine()
        self.session = sessionmaker(bind=self.engine)()

    def create_engine(self):
        assert self.group in ["detection", "face"]
        from common_config import get_database
        database = get_database(self.group)
        engine = create_engine(database)
        return engine

    def create_tables(self):
        Base.metadata.create_all(self.engine)
        logger.info("created all tables.")

    def exist(self, table, column, key):
        session = self.session
        query = session.query(table).filter(table.get_column(column).in_([key])).all()
        return len(query) > 0

    @staticmethod
    def interaction(table, key):
        table_name = table.__tablename__
        attention = "key {} doesnt exists in table {}, are you sure to add it? y/n:".format(key, table_name)
        ret = input(attention)
        return ret

    def get_id(self, table, column, key):
        query = self.session.query(table).filter(table.get_column(column) == key).first().id
        return query

    def ensure_customer_id(self, name, customer_type):
        if self.exist(Customer, "name", name):
            return self.get_id(Customer, "name", name)
        else:

            assert name.isupper()

            ret = self.interaction(Customer, name)
            assert ret.upper() == "Y"
            new_item = Customer(name=name, customer_type=customer_type)
            self.add_item(new_item)
            return new_item.id

    def ensure_scene_id(self, name, customer_id):

        if self.exist(Scene, "name", name):
            return self.get_id(Scene, "name", name)
        else:
            assert len(name.split("-")) == 2
            assert (name.split("-")[0].islower() and name.split("-")[1].islower())

            ret = self.interaction(Scene, name)
            assert ret.upper() == "Y"

            new_item = Scene(name=name, customer_id=customer_id)
            self.add_item(new_item)
            return new_item.id

    def ensure_channel_id(self, key):
        if self.exist(Channel, "name", key):
            return self.get_id(Channel, "name", key)
        else:

            # ret = self.interaction(Channel, key)
            # assert ret.upper() == "Y"

            new_item = Channel(name=key)
            self.add_item(new_item)
            return new_item.id

    def ensure_label_id(self, key):
        if self.exist(DetLabel, "name", key):
            return self.get_id(DetLabel, "name", key)
        else:

            ret = self.interaction(DetLabel, key)
            assert ret.upper() == "Y"

            new_item = DetLabel(name=key)

            self.add_item(new_item)
            return new_item.id

    def add_item(self, item):
        self.session.add(item)
        self.session.commit()
        logger.info("added and commit a new item in table {}".format(item.__tablename__))

    def commit_all(self):
        self.session.commit()
        logger.info("all items committed.")

    def close_all_session(self):
        self.session.close()
        self.engine.dispose()
        logger.info("all sessions and engine are closed")

    # todo
    def overwrite(self):
        pass

    # todo
    def remove_item(self):
        pass
