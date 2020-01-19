import glob
import hashlib
import json
import logging
import os
import tarfile
import time
from multiprocessing import Pool

from python_common.hdfs.hdfsCli import HdfsClient
from tqdm import tqdm

from db_lib import DBClient
from db_lib import DetData, FaceData, Customer, Scene, Channel, DetLabel, DetValData

logger = logging.getLogger()


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class DataSelector(object):
    def __init__(self, config, num_process):

        super(DataSelector, self).__init__()
        self.data = config["data"]
        self.group = self.data["group"]
        self.train_data_config = config["train_data_config"]
        self.label_config = config["label_config"]
        self.customer_config = config["customer_config"]
        self.scene_config = config["scene_config"]
        self.channel_config = config["channel_config"]
        self.val_data_config = config["val_data_config"]
        self.num_process = num_process
        self.db_client = DBClient()
        self.valid_configs, self.tables = self.get_valid_args_tables()

        self.db_client = DBClient()

    def get_valid_args_tables(self):
        valid_configs = {}
        tables = {}
        assert not os.path.exists(self.data["tmp_root"])
        os.makedirs(self.data["tmp_root"])
        logger.info("makedirs tmp_root: {}".format(self.data["tmp_root"]))
        assert not os.path.exists(self.data["dst_root"])
        os.makedirs(self.data["dst_root"])
        logger.info("makedirs dst_root: {}".format(self.data["dst_root"]))

        assert self.group in ["detection", "face"]

        assert self.train_data_config

        if self.group == "detection":
            assert len(self.train_data_config["label_type"]) == 1

        for cur_key in self.train_data_config:
            assert isinstance(self.train_data_config[cur_key], list), "{} should be list".format(cur_key)

        valid_configs["data"] = self.train_data_config
        tables["data"] = DetData if self.group == "detection" else FaceData

        if self.group == "detection":
            assert self.label_config
            assert "name" in self.label_config

            for cur_key in self.label_config:
                assert isinstance(self.label_config[cur_key], list), "{} should be list".format(cur_key)

            valid_configs["label"] = self.label_config
            tables["label"] = DetLabel

        if self.customer_config:
            for cur_key in self.customer_config:
                assert isinstance(self.customer_config[cur_key], list), "{} should be list".format(cur_key)

            valid_configs["customer"] = self.customer_config
            tables["customer"] = Customer
            tables["scene"] = Scene

        if self.scene_config:
            for cur_key in self.scene_config:
                assert isinstance(self.scene_config[cur_key], list), "{} should be list".format(cur_key)

            valid_configs["scene"] = self.scene_config
            tables["scene"] = Scene

        if self.channel_config and self.group == "detection":
            for cur_key in self.scene_config:
                assert isinstance(self.scene_config[cur_key], list), "{} should be list".format(cur_key)

            valid_configs["channel"] = self.channel_config
            tables["channel"] = Channel

        logger.info("valid_configs: {}".format(valid_configs))

        return valid_configs, tables

    def get_property(self, name):
        return self.__getattribute__(name)

    def get_train_paths(self):
        valid_configs = self.valid_configs
        session = self.db_client.session
        tables = self.tables
        final_ids = set()
        for cur_config_key in valid_configs:
            cur_config = valid_configs[cur_config_key]
            cur_table = tables[cur_config_key]
            for key in cur_config:
                if cur_table == tables["customer"]:
                    queries = session.query(cur_table).filter(cur_table.get_column(key).in_(cur_config[key])).all()
                    customer_ids = [query.id for query in queries]
                    queries = session.query(tables["scene"]).filter(tables["scene"].customer_id.in_(customer_ids)).all()
                    scene_ids = [query.id for query in queries]
                    queries = session.query(tables["data"]).filter(tables["data"].scene_id.in_(scene_ids)).all()
                elif cur_table == tables["data"]:
                    queries = session.query(cur_table).filter(cur_table.get_column(key).in_(cur_config[key])).all()
                else:  # ["scene", "label", "channel"]
                    queries = session.query(cur_table).filter(cur_table.get_column(key).in_(cur_config[key])).all()
                    ids_in_datatable = [query.id for query in queries]
                    if "scene" in tables and cur_table == tables["scene"]:
                        new_key = "scene_id"
                    elif "channel" in tables and cur_table == tables["channel"]:
                        new_key = "channel_id"
                    elif "label" in tables and cur_table == tables["label"]:
                        new_key = "label_id"
                    else:
                        raise NotImplementedError("only three tables support")

                    queries = session.query(tables["data"]).filter(
                        tables["data"].get_column(new_key).in_(ids_in_datatable)).all()

                assert queries, "can not find any data in data_table for current key: {} \n" \
                                "please check your query".format(key)
                cur_ids = [query.id for query in queries]
                if not final_ids:
                    final_ids = set(cur_ids)
                else:
                    final_ids = final_ids & set(cur_ids)

        queries = session.query(tables["data"]).filter(tables["data"].id.in_(final_ids)).all()
        assert len(queries) > 0

        download_train_list = [query.to_dict() for query in queries]

        return download_train_list

    def get_val_paths(self):

        val_data_config = self.val_data_config
        session = self.db_client.session
        final_ids = set()
        logger.info("val data config : {}".format(val_data_config))
        if "id" in val_data_config and "unique_key" in val_data_config:
            raise NotImplementedError("only support one key")

        if "id" in val_data_config or "unique_key" in val_data_config:
            for key in val_data_config:
                assert isinstance(val_data_config[key], list), "config value should be list"
                assert len(val_data_config[key]) == 1, "val data length should be 1"
                queries = session.query(DetValData).filter(DetValData.get_column(key).in_(val_data_config[key])).all()
                assert queries, "can not find any data in data_table for current key: {} \n" \
                                "please check your query".format(key)

                cur_ids = [query.id for query in queries]
                if not final_ids:
                    final_ids = set(cur_ids)
                else:
                    final_ids = final_ids & set(cur_ids)

            queries = session.query(DetValData).filter(DetValData.id.in_(final_ids)).all()
            assert len(queries) == 1, "wrong queries"

            download_val_list = [query.to_dict() for query in queries]

            return download_val_list

        else:
            raise NotImplementedError("only support id and unique_key to query")

    @staticmethod
    def download_single(hdfs_path, tmp_root, dst_root=None):
        new_client = HdfsClient()
        logger.info(hdfs_path)
        assert new_client.exist(hdfs_path)
        if hdfs_path.endswith(".tar"):
            logger.debug("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            logger.debug("hdfs_path {} --> dst_root: {}".format(hdfs_path, dst_root))
            logger.debug("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            assert dst_root
            new_client.download(hdfs_path, tmp_root)
            logger.info("download {} --> {}".format(hdfs_path, tmp_root))
            cur_tar = tarfile.open(os.path.join(tmp_root, hdfs_path.split("/")[-1]), "r:")
            for m in cur_tar.getmembers():
                m.path = m.path.split('/')[-1]

            cur_tar.extractall(dst_root)

            cur_tar.close()
            logger.info("extract {} --> {}".format(hdfs_path.split("/")[-1], dst_root))

        elif hdfs_path.endswith(".json"):
            new_client.download(hdfs_path, tmp_root)
            logger.info("download {} --> {}".format(hdfs_path, tmp_root))
        else:
            raise NotImplementedError

    @staticmethod
    def merge_det_anno(labels, anno_path, anno_hdfs_paths, label_type):
        logger.info("merge annotations...")
        assert len(label_type) == 1
        label_type = label_type[0]

        # TODO how to handle keypoint ???
        if label_type == "keypoint":
            raise NotImplementedError("")
        elif label_type in ["mask", "bbox"]:
            categories = [{"id": index + 1, "name": label} for index, label in enumerate(labels)]
        else:
            raise NotImplementedError("")

        logger.info("merged categories --> {}".format(categories))

        label_dict = {}
        for category in categories:
            label_dict[category["name"]] = category["id"]

        logger.info("label dict: {}".format(label_dict))

        all_anno_json = {'images': [], 'annotations': [], 'categories': categories}
        global_annos_id = 1
        images, annotations = [], []

        # merge
        for index, cur_file in enumerate(anno_hdfs_paths):
            cur_abs_path = os.path.join(anno_path, cur_file.split("/")[-1])
            cur_label = cur_file.split(".json")[0].split("_")[-1]

            with open(cur_abs_path, "r") as f:
                cur_anno = json.load(f)

            # int id
            images += cur_anno["images"]
            for an in cur_anno["annotations"]:
                an["category_id"] = label_dict[cur_label]
                an["id"] = global_annos_id
                global_annos_id += 1

            annotations += cur_anno["annotations"]
            logger.info("The {} file {} merged!".format(index + 1, cur_abs_path))

        all_anno_json["images"] = images
        all_anno_json["annotations"] = annotations
        logger.info("merged all!")
        return all_anno_json

    def download_train(self, download_train_list, num_process):

        hdfs_data_paths = set([single["hdfs_path"] for single in download_train_list])
        hdfs_anno_paths = set([single["hdfs_anno_path"] for single in download_train_list])
        from functools import partial
        tmp_root = self.data["tmp_root"]
        dst_root = self.data["dst_root"]
        train_data_dst_root = os.path.join(dst_root, "train")
        train_anno_dst_root = os.path.join(dst_root, "annotations")
        if not os.path.exists(train_data_dst_root):
            os.makedirs(train_data_dst_root)
            logger.info("mkdir train_data_dst_root: {}".format(train_data_dst_root))
        if not os.path.exists(train_anno_dst_root):
            os.makedirs(train_anno_dst_root)
            logger.info("mkdir train_anno_dst_root: {}".format(train_anno_dst_root))

        pool = Pool(processes=num_process)
        logger.debug("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info("downloading train data --> {}...".format(tmp_root))
        pool.map(partial(self.download_single, dst_root=train_data_dst_root, tmp_root=tmp_root), hdfs_data_paths)
        logger.debug("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        logger.debug("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info("downloading train annotations --> {}".format(tmp_root))
        logger.info(hdfs_anno_paths)
        pool.map(partial(self.download_single, tmp_root=tmp_root), hdfs_anno_paths)
        logger.debug("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        if self.group == "detection":
            labels = self.label_config["name"]
            label_type = self.train_data_config["label_type"]
            merged_data = self.merge_det_anno(labels, tmp_root, hdfs_anno_paths, label_type)
            with open(os.path.join(train_anno_dst_root, "instances_train.json"), "w") as f:
                json.dump(merged_data, f)
            logger.info(
                "merged train annotations --> {}".format(os.path.join(train_anno_dst_root, "instances_train.json")))
        else:
            raise NotImplementedError("")

    def download_val(self, download_val_list):
        hdfs_data_paths = [single["hdfs_path"] for single in download_val_list]
        hdfs_anno_paths = [single["hdfs_anno_path"] for single in download_val_list]
        assert len(hdfs_anno_paths) == 1
        assert len(hdfs_anno_paths) == 1
        tmp_root = self.data["tmp_root"]
        dst_root = self.data["dst_root"]
        val_data_dst_root = os.path.join(dst_root, "val")
        val_anno_dst_root = os.path.join(dst_root, "annotations")
        if not os.path.exists(val_data_dst_root):
            os.makedirs(val_data_dst_root)
            logger.info("mkdir val_data_dst_root: {}".format(val_data_dst_root))
        if not os.path.exists(val_anno_dst_root):
            os.makedirs(val_anno_dst_root)
            logger.info("mkdir val_anno_dst_root: {}".format(val_anno_dst_root))

        self.download_single(hdfs_data_paths[0], tmp_root=tmp_root, dst_root=val_data_dst_root)

        self.download_single(hdfs_anno_paths[0], tmp_root=val_anno_dst_root)

        os.rename(os.path.join(val_anno_dst_root, hdfs_anno_paths[0].split("/")[-1]),
                  os.path.join(val_anno_dst_root, "instances_val.json"))

        logger.info("rename val annotations --> {}".format(os.path.join(val_anno_dst_root, "instances_val.json")))

    def run(self):
        # 1 select 2 download

        num_process = self.num_process
        download_list = []
        download_train_list = self.get_train_paths()
        download_list += download_train_list
        self.download_train(download_train_list, num_process)
        if self.val_data_config:
            download_val_list = self.get_val_paths()
            download_list += download_val_list

            self.download_val(download_val_list)
        from common_config import donwload_local_file
        json.dump(download_list, open(donwload_local_file, "w"))


class DataBuilder(object):

    def __init__(self, config, num_process=1):
        super(DataBuilder, self).__init__()
        self.config = config
        self.data = config["data"]

        assert self.data["type"] in ["train", "val"]

        if self.data["type"] == "val":
            self.val_data_config = config["val_data_config"]
        else:
            self.train_data_config = config["train_data_config"]  # {"camera": ["did"], "label_type": ["bbox"]}
            self.customer_config = config["customer_config"]
            self.scene_config = config["scene_config"]
            # self.val_data_config = config["val_data_config"]
            self.channel_config = config["channel_config"]
            self.label_config = config["label_config"]

        self.num_process = num_process
        self.build_single_file = True if self.data["type"] == "val" else self.data["build_single_file"]

        self.valid_configs, self.tables = self.get_valid_args_tables()

        self.ensure_anno()
        self.hdfs_data_root, self.hdfs_anno_root = self.get_upload_hadoop_root()
        self.db_client = DBClient()
        self.hdfs_client = HdfsClient()

    def get_upload_hadoop_root(self):
        if self.data["type"] == "val":
            from common_config import hdfs_val_anno_dir, hdfs_val_anno_dir
            hdfs_data_root = hdfs_val_anno_dir
            hdfs_anno_root = hdfs_val_anno_dir
        else:
            config = self.config
            from common_config import hdfs_train_anno_dir, hdfs_train_data_dir
            camera = config["train_data_config"]["camera"][0]
            customer = config["customer_config"]["name"][0]
            scene = config["scene_config"]["name"][0]
            hdfs_data_root = hdfs_train_data_dir.format(camera, customer, scene)
            hdfs_anno_root = hdfs_train_anno_dir.format(camera, customer, scene)

        return hdfs_data_root, hdfs_anno_root

    def get_valid_args_tables(self):
        valid_configs = {}
        tables = {}

        assert os.path.exists(self.data["image_path"])
        assert os.path.isfile(self.data["anno_path"])
        assert not os.path.exists(self.data["tmp_root"])
        os.makedirs(self.data["tmp_root"])

        if self.data["type"] == "val":
            assert self.val_data_config
            assert len(self.val_data_config["labels"]) == 1
            assert len(self.val_data_config["unique_key"]) == 1
            assert len(self.val_data_config["label_type"]) == 1
            assert len(self.val_data_config["camera"]) == 1

            valid_configs["data"] = self.val_data_config
            tables["data"] = DetValData

        else:
            assert self.scene_config
            assert len(self.scene_config["name"]) == 1
            valid_configs["scene"] = self.scene_config

            tables["scene"] = Scene
            tables["customer"] = Customer

            assert self.customer_config
            assert len(self.customer_config["customer_type"]) == 1
            assert len(self.customer_config["name"]) == 1
            valid_configs["customer"] = self.customer_config

            assert self.train_data_config
            assert len(self.train_data_config["camera"]) == 1  # did fid fisheye
            assert len(self.train_data_config["label_type"]) == 1  # bbox keypoint ...
            valid_configs["data"] = self.train_data_config
            tables["data"] = DetData

            assert self.label_config
            logger.debug(self.label_config["name"])
            logger.debug(type(self.label_config["name"]))
            assert isinstance(self.label_config["name"], list)  # body head face

            valid_configs["label"] = self.label_config
            tables["label"] = DetLabel

            if self.build_single_file:
                self.channel_config["name"] = self.channel_config["name"] if self.channel_config["name"] else "ch00000"
                valid_configs["channel"] = self.channel_config
                tables["channel"] = Channel

        # TODO handle keypoint and mask
        if valid_configs["data"]["label_type"][0] != "bbox":
            raise NotImplementedError("only support bbox now")

        logger.info("valid configs: {}".format(valid_configs))

        return valid_configs, tables

    @staticmethod
    def check_channel_date(img_name):
        assert "ch" in img_name
        ch_index = img_name.index("ch")
        ch_date_candidate = img_name[ch_index: ch_index + len("ch01001_20200101")]
        assert "_" in ch_date_candidate
        date_candidate = ch_date_candidate.split("_")[1]
        assert len(date_candidate) == len("20200101")
        try:
            time.strptime(date_candidate, "%Y%m%d")
        except:
            raise Exception("can not get channel_date from {}".format(img_name))

    def ensure_anno(self):
        logger.info("check annotations format...")
        anno_data = json.load(open(self.data["anno_path"]))
        img_path = self.data["image_path"]

        if self.data["type"] == "train":
            labels = self.label_config["name"]
            # ensure label
            logger.info("check categories ...")
            assert len(labels) == len(anno_data["categories"])
            for index, category in enumerate(anno_data["categories"]):
                assert labels[index] == category["name"]
                assert category["id"] == index + 1

        # ensure int id
        logger.info("check image_id and annotations...")
        img_ids = [img["id"] for img in anno_data["images"]]
        for cur_id in img_ids:
            if not isinstance(cur_id, int):
                raise TypeError("img id should be integer")

        # ensure images and annotations
        img_files = set(os.listdir(img_path))
        id2img = dict()
        for img in anno_data["images"]:
            img_name = img["file_name"]
            id2img[img["id"]] = img_name
            if not self.build_single_file:
                self.check_channel_date(img_name)
        for an in anno_data["annotations"]:
            assert an["image_id"] in id2img
            assert id2img[an["image_id"]] in img_files

    @staticmethod
    def build_single_anno(json_data, dst_anno):
        assert not os.path.exists(dst_anno)
        with open(dst_anno, "w") as f:
            json.dump(json_data, f)
        logger.info("json file {} built.".format(dst_anno))

    @staticmethod
    def build_single(imgs, dst_name, show_schedule=False):
        tar_file = tarfile.open(dst_name, "w")
        second_folder = dst_name.split("/")[-1].split(".tar")[0]

        if show_schedule:
            for cur_file in tqdm(imgs):
                img_file_name = cur_file.split("/")[-1]
                tar_file.add(cur_file, arcname="./{}/{}".format(second_folder, img_file_name))
        else:
            for cur_file in imgs:
                img_file_name = cur_file.split("/")[-1]
                tar_file.add(cur_file, arcname="./{}/{}".format(second_folder, img_file_name))

        tar_file.close()
        logger.info("tarfile {} built.".format(dst_name))

    @staticmethod
    def interface_to_build_multi(inputs_args_list):
        imgs = inputs_args_list[0]
        dst_name = inputs_args_list[1]

        tar_file = tarfile.open(dst_name, "w")
        second_folder = dst_name.split("/")[-1].split(".tar")[0]

        for cur_file in imgs:
            img_file_name = cur_file.split("/")[-1]
            tar_file.add(cur_file, arcname="./{}/{}".format(second_folder, img_file_name))

        tar_file.close()
        logger.info("tarfile {} built.".format(dst_name))

    @staticmethod
    def interface_to_build_multi_anno(inputs_args_list):
        json_data = inputs_args_list[0]
        dst_anno = inputs_args_list[1]
        assert not os.path.exists(dst_anno)
        with open(dst_anno, "w") as f:
            json.dump(json_data, f)
        logger.info("json file {} built.".format(dst_anno))

    def build_multi(self, img_path, anno_path, tmp_root, labels, num_process):
        imgs = os.listdir(img_path)
        channel_date_dict = {}
        for img in imgs:
            ch_index = img.index("ch")
            ch_date = img[ch_index: ch_index + len("ch01001_20200101")]
            if ch_date not in channel_date_dict:
                channel_date_dict[ch_date] = [os.path.join(img_path, img)]
            else:
                channel_date_dict[ch_date].append(os.path.join(img_path, img))

        inputs_list = []
        for channel_date in channel_date_dict:
            cur_dst_name = os.path.join(tmp_root, "{}.tar".format(channel_date))
            inputs_list.append([channel_date_dict[channel_date], cur_dst_name])

        pool = Pool(num_process)
        logger.info("building tars...")
        pool.map(self.interface_to_build_multi, inputs_list)

        json_data_dict = {}
        """
        {"ch00000_20200101": 
            {"body": {"images":[], "annotations": [], "categories": []}, 
             "head": {...},
             "face": {...} 
            }
        }
        """
        logger.info("split {} ...".format(anno_path))
        json_data = json.load(open(anno_path, "r"))
        id_to_imgname = {}
        id_to_img = {}
        for img_info in json_data["images"]:
            id_to_imgname[img_info["id"]] = img_info["file_name"]
            id_to_img[img_info["id"]] = img_info

        id_to_label = {}
        for index, label in enumerate(labels):
            id_to_label[index + 1] = label

        for an in json_data["annotations"]:
            img = id_to_img[an["image_id"]]  # {"file_name": "dsfds", "height":1920, "width":1920, "id": 0}
            imgname = id_to_imgname[an["image_id"]]  #
            label = id_to_label[an["category_id"]]
            ch_index = imgname.index("ch")
            ch_date = imgname[ch_index: ch_index + len("ch01001_20200101")]
            an["category_id"] = 1
            if ch_date in json_data_dict:
                if label in json_data_dict[ch_date]:
                    json_data_dict[ch_date][label]["images"].append(img)
                    json_data_dict[ch_date][label]["annotations"].append(an)
                else:
                    json_data_dict[ch_date][label] = {"images": [img],
                                                      "annotations": [an],
                                                      "categories": [{"id": 1, "name": label}]}
            else:
                json_data_dict[ch_date] = {}
                json_data_dict[ch_date][label] = {"images": [img],
                                                  "annotations": [an],
                                                  "categories": [{"id": 1, "name": label}]}

        logger.info("finished split, all channel date: {}".format(json_data_dict.keys()))
        anno_inputs_list = []
        for channel_date in json_data_dict:
            for label in json_data_dict[channel_date]:
                cur_dst_name = os.path.join(tmp_root, "{}_{}.json".format(channel_date, label))
                anno_inputs_list.append([json_data_dict[channel_date][label], cur_dst_name])

        logger.info("building jsons...")
        pool.map(self.interface_to_build_multi_anno, anno_inputs_list)

    def build_val_data(self):
        img_path = self.data["image_path"]
        anno_path = self.data["anno_path"]
        tmp_root = self.data["tmp_root"]
        anno_md5 = md5(anno_path)
        imgs = glob.glob("{}/*.jpg".format(img_path))
        labels = self.valid_configs["data"]["labels"][0]

        dst_tar = "whitebox_{}_{}.tar".format(len(imgs), anno_md5)
        dst_anno = "whitebox_{}_{}_{}.json".format(len(imgs), labels, anno_md5)

        logger.info("build val data --> {}".format(os.path.join(tmp_root, dst_tar)))
        self.build_single(imgs, os.path.join(tmp_root, dst_tar), show_schedule=True)
        json_data = json.load(open(anno_path, "r"))
        logger.info("build val anno --> {}".format(os.path.join(tmp_root, dst_anno)))
        self.build_single_anno(json_data, os.path.join(tmp_root, dst_anno))

    def build_train_data(self, num_process=1):
        img_path = self.data["image_path"]
        anno_path = self.data["anno_path"]
        tmp_root = self.data["tmp_root"]
        labels = self.valid_configs["label"]["name"]
        anno_md5 = md5(anno_path)
        imgs = glob.glob("{}/*.jpg".format(img_path))

        if self.build_single_file:
            # all_GENERAL_general-general_imgnum_hd5.tar
            customer_type = self.customer_config["customer_type"][0]
            customer_name = self.customer_config["name"][0]
            scene_name = self.customer_config["name"][0]
            dst_tar = "{}_{}_{}_{}_{}.tar".format(customer_type,
                                                  customer_name,
                                                  scene_name,
                                                  len(imgs),
                                                  anno_md5)
            logger.info("build train data --> {}".format(os.path.join(tmp_root, dst_tar)))

            self.build_single(imgs, os.path.join(tmp_root, dst_tar), show_schedule=True)
            json_data = json.load(open(anno_path, "r"))

            logger.info("split train annotations ...")

            for index, cur_label in enumerate(labels):
                new_annos = []
                # all_GENERAL_general-general_imgnum_hd5_body.json
                dst_anno = "{}_{}_{}_{}_{}_{}.json".format(customer_type,
                                                           customer_name,
                                                           scene_name,
                                                           len(imgs),
                                                           anno_md5,
                                                           cur_label)
                for an in json_data["annotations"]:
                    if an["category_id"] == index + 1:
                        an["category_id"] = 1  # reset to 1
                        new_annos.append(an)

                new_json_data = {"images": json_data["images"],
                                 "annotations": new_annos,
                                 "categories": [{"id": 1, "name": cur_label}]}
                logger.info("split train annotations --> {}".format(os.path.join(tmp_root, dst_anno)))
                self.build_single_anno(new_json_data, os.path.join(tmp_root, dst_anno))

        else:
            #  build multi
            self.build_multi(img_path, anno_path, tmp_root, labels, num_process)

    def build(self, num_process):
        if self.data["type"] == "val":
            logger.info("build val data ...")
            self.build_val_data()
        else:
            logger.info("build train data ...")
            self.build_train_data(num_process=num_process)

    @staticmethod
    def upload_data(inputs):
        src = inputs[0]
        dst = inputs[1]
        logger.debug(src)
        logger.debug(dst)
        # finish
        logger.debug("newing a client ...")
        new_client = HdfsClient()  # default bj
        logger.debug("newed")
        new_client.upload(dst, src)
        logger.info("upload {} --> {}".format(src, dst))

    def upload(self, num_process):
        tmp_root = self.data["tmp_root"]
        built_input = []  # [[src, dst],...,[]]
        hdfs_client = self.hdfs_client
        hdfs_data_root = self.hdfs_data_root
        hdfs_anno_root = self.hdfs_anno_root

        if not hdfs_client.exist(hdfs_data_root):
            hdfs_client.makedirs(hdfs_data_root, permission=777)
        if not hdfs_client.exist(hdfs_anno_root):
            hdfs_client.makedirs(hdfs_anno_root, permission=777)

        logger.info("hdfs_data_root: {}".format(hdfs_data_root))
        logger.info("hdfs_anno_root: {}".format(hdfs_anno_root))

        file_list = os.listdir(tmp_root)
        for cur_file in file_list:
            if ".json" in cur_file:
                built_input.append([os.path.join(tmp_root, cur_file), hdfs_anno_root])
                if hdfs_client.exist(os.path.join(hdfs_anno_root, cur_file)):
                    raise AssertionError("{} exists".format(os.path.join(hdfs_anno_root, cur_file)))
            elif ".tar" in cur_file:
                built_input.append([os.path.join(tmp_root, cur_file), hdfs_data_root])
                if hdfs_client.exist(os.path.join(hdfs_data_root, cur_file)):
                    raise AssertionError("{} exists".format(os.path.join(hdfs_data_root, cur_file)))
            else:
                raise NotImplementedError("")

        pool = Pool(num_process)
        pool.map(self.upload_data, built_input)

    @staticmethod
    def get_tar_num(tar_file):
        assert tar_file.endswith(".tar")
        tar = tarfile.open(tar_file, "r")
        tar_num = len(tar.getmembers())
        tar.close()
        return tar_num

    def add_to_db(self):
        camera = self.valid_configs["data"]["camera"][0]
        label_type = self.valid_configs["data"]["label_type"][0]
        tmp_root = self.data["tmp_root"]
        db_client = self.db_client

        hdfs_data_root = self.hdfs_data_root
        hdfs_anno_root = self.hdfs_anno_root

        upload_list = []
        total_item = 0
        if self.data["type"] == "val":
            logger.info("add val data to db ...")
            unique_key = self.valid_configs["data"]["unique_key"]
            labels = self.valid_configs["data"]["labels"]
            json_list = [cur_file for cur_file in os.listdir(tmp_root) if ".json" in cur_file]
            tar_list = [cur_file for cur_file in os.listdir(tmp_root) if ".tar" in cur_file]
            img_num = self.get_tar_num(os.path.join(tmp_root, tar_list[0]))
            logger.debug(img_num)
            assert len(json_list) == 1
            assert len(tar_list) == 1
            data_dst_path = os.path.join(hdfs_data_root, tar_list[0])
            logger.debug(data_dst_path)

            anno_dst_path = os.path.join(hdfs_anno_root, json_list[0])
            logger.debug(anno_dst_path)

            new_item = DetValData(labels=labels,
                                  label_type=label_type,
                                  camera=camera,
                                  unique_key=unique_key,
                                  img_num=img_num,
                                  hdfs_path=data_dst_path,
                                  hdfs_anno_path=anno_dst_path)

            upload_list.append(new_item.to_dict())
            db_client.add_item(new_item)
            total_item += 1
        else:
            customer = self.valid_configs["customer"]["name"][0]
            customer_type = self.valid_configs["customer"]["customer_type"][0]
            scene = self.valid_configs["scene"]["name"][0]
            logger.info("add train data to db ...")
            customer_id = db_client.ensure_customer_id(name=customer, customer_type=customer_type)
            scene_id = db_client.ensure_scene_id(name=scene, customer_id=customer_id)
            json_list = [cur_file for cur_file in os.listdir(tmp_root) if ".json" in cur_file]
            if self.build_single_file:
                tar_list = [cur_file for cur_file in os.listdir(tmp_root) if ".tar" in cur_file]
                assert len(tar_list) == 1
                assert len(json_list) == len(self.valid_configs["label"]["name"])
                data_dst_path = os.path.join(hdfs_data_root, tar_list[0])
                img_num = self.get_tar_num(os.path.join(tmp_root, tar_list[0]))
                for json_file in json_list:
                    anno_dst_path = os.path.join(hdfs_anno_root, json_file)
                    cur_label = json_file.split(".json")[0].split("_")[-1]
                    label_id = db_client.ensure_label_id(cur_label)
                    channel_id = db_client.ensure_channel_id(self.valid_configs["channel"]["name"])
                    new_item = DetData(scene_id=scene_id,
                                       channel_id=channel_id,
                                       camera=camera,
                                       hdfs_path=data_dst_path,
                                       hdfs_anno_path=anno_dst_path,
                                       img_num=img_num,
                                       label_id=label_id,
                                       label_type=label_type)

                    db_client.add_item(new_item)
                    upload_list.append(new_item.to_dict())

                    total_item += 1
            else:
                for json_file in json_list:
                    anno_dst_path = os.path.join(hdfs_anno_root, json_file)
                    tar_file_name = str(json_file.split("_")[0]) + "_" + str(json_file.split("_")[1]) + ".tar"
                    assert os.path.exists(os.path.join(tmp_root, tar_file_name))
                    data_dst_path = os.path.join(hdfs_data_root, tar_file_name)
                    img_num = self.get_tar_num(os.path.join(tmp_root, tar_file_name))
                    cur_label = json_file.split(".json")[0].split("_")[-1]
                    label_id = db_client.ensure_label_id(cur_label)
                    channel_id = db_client.ensure_channel_id(json_file.split("_")[0])

                    new_item = DetData(scene_id=scene_id,
                                       channel_id=channel_id,
                                       camera=camera,
                                       hdfs_path=data_dst_path,
                                       hdfs_anno_path=anno_dst_path,
                                       img_num=img_num,
                                       label_id=label_id,
                                       label_type=label_type)

                    db_client.add_item(new_item)
                    upload_list.append(new_item.to_dict())
                    total_item += 1

        logger.info("add all items to dbs, total {} new items".format(total_item))
        db_client.commit_all()
        db_client.close_all_session()
        from common_config import upload_local_file
        logger.info("saving upload list log --> {}".format(upload_local_file))
        json.dump(upload_list, open(upload_local_file, "w"))
        logger.info("{} saved".format(upload_local_file))

    def run(self):
        num_process = self.num_process
        # todo 1 build 2 upload 3 add to db ???
        self.build(num_process=num_process)
        self.upload(num_process=num_process)
        self.add_to_db()


class DataDeltor(object):
    # TODO add DataDeltor, use upload or download log to delete
    def __init__(self):
        # TODO use upload or download log to delete
        pass
