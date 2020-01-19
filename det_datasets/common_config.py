# -*- coding:utf-8 -*-
# @Author:      likuan kli@aibee.com
# @Time:        2020-01-18 00:30
# @FILE_NAME:   common_config.py


hdfs_val_data_dir = "/staging/data/detection/val/images/"
hdfs_val_anno_dir = "/staging/data/detection/val/annotations/"
hdfs_train_data_dir = "/staging/data/detection/train/{}/{}/{}/images/"  # did GENERAL general-general
hdfs_train_anno_dir = "/staging/data/detection/train/{}/{}/{}/annotations/"

# backup info
backup_dir = "/staging/data/detection/backup"
backup_list = "/staging/data/detection/backup.list"

# download log
download_log_dir = "/staging/data/detection/download/"
donwload_local_file = "download.list"
# upload log
upload_log_dir = "/staging/data/detection/upload/"
upload_local_file = "upload.list"

# db info
host = "172.20.10.20"
port = 3306
user = "user"
password = "password"
det_datasets = "test_det"
face_datasets = ""


def get_backup_path():
    return backup_list, backup_dir


def get_database(group):
    if group == "detection":
        return "mysql+pymysql://{}:{}@{}:{}/{}".format(user, password, host, port, det_datasets)
    elif group == "face":
        raise NotImplementedError("face not implemented")
    else:
        raise NotImplementedError("")


def get_log_path(mode):
    assert mode in ["download", "upload"]
    hadoop_log_dir = download_log_dir if mode == "download" else upload_log_dir
    local_file = donwload_local_file if mode == "download" else upload_local_file
    return local_file, hadoop_log_dir


def get_backup_cmd(backup_file):
    backup_cmd = "mysqldump -h{} -P{} -u{} -p{} {} > {}".format(host, port, user,
                                                                password, det_datasets, backup_file)

    return backup_cmd


def get_recovery_cmd(backup_file):
    recovery_cmd = "mysql -h{} -P{} -u{} -p{} {} < {}".format(host, port, user,
                                                              password, det_datasets, backup_file)

    return recovery_cmd
