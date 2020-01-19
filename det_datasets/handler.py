# -*- coding:utf-8 -*-
# @Author:      likuan kli@aibee.com
# @Time:        2020-01-19 17:15
# @FILE_NAME:   handler.py

import logging
import multiprocessing
import os
import time

import yaml
from python_common.hdfs.hdfsCli import HdfsClient

from data_lib import DataBuilder, DataSelector

logger = logging.getLogger()


class Handler(object):

    def __init__(self, config_file, num_process=-1):
        self.mode, self.valid_config, self.owner = self.read_config(config_file)
        self.num_process = num_process if num_process > 0 else multiprocessing.cpu_count()
        self.client = HdfsClient()

        # TODO add DataDeltor, use upload or download log to delete

        self.data_handler = DataBuilder(self.valid_config, self.num_process) \
            if self.mode == "upload" else DataSelector(self.valid_config, self.num_process)
        logger.info("handler init successful.")

    @staticmethod
    def read_config(config_file):
        assert os.path.exists(config_file)
        logger.info("reading config file {}".format(config_file))
        with open(config_file) as f:
            config_data = yaml.load(f)
            assert "mode" in config_data
            mode = config_data["mode"]
            assert mode in ["download", "upload"]
            assert config_data["owner"]
            owner = config_data["owner"]
            config_data = config_data[mode]
        logger.info("mode: {}, owner: {}, config_data: {}".format(mode, owner, config_data))
        return mode, config_data, owner

    def backup_database(self, backup_time):
        from common_config import get_backup_path
        from common_config import get_backup_cmd
        owner = self.owner
        client = self.client
        backup_list, backup_dir = get_backup_path()
        start_sql_cmd = "service mysql start"
        logger.info("start mysql service")
        os.system(start_sql_cmd)
        back_file_name = "{}_{}_bak.sql".format(backup_time, owner)
        backup_cmd = get_backup_cmd(back_file_name)
        logger.info("run back cmd: {}".format(backup_cmd))
        os.system(backup_cmd)

        if not client.exist(backup_dir):
            client.makedirs(backup_dir, permission=777)
            logger.info("{} not exists, created!".format(backup_dir))

        if not client.exist(backup_list):
            client.write(backup_list, data="")
            logger.info("{} not exists, created!".format(backup_list))

        client.upload(backup_dir, back_file_name)
        logger.info("upload {} --> {}".format(back_file_name, backup_dir))
        client.write(backup_list, data="\n{}".format(back_file_name), append=True)

    def recovery(self, backup_file):
        # TODO recovery from back.sql
        from common_config import get_recovery_cmd
        recovery_cmd = get_recovery_cmd(backup_file)
        raise NotImplementedError()

    def update_history_logs(self, update_time):
        from common_config import get_log_path
        mode, owner = self.mode, self.owner
        local_file, hadoop_log_dir = get_log_path(mode)
        logger.info("get local file: {}, hadoop_log_dir: {}".format(local_file, hadoop_log_dir))
        client = self.client
        if not client.exist(hadoop_log_dir):
            client.makedirs(hadoop_log_dir)
            logger.info("{} not exists, created!".format(hadoop_log_dir))
        hadoop_des_filename = os.path.join(hadoop_log_dir, "{}_{}_{}".format(update_time, owner, local_file))
        client.upload(hadoop_des_filename, local_file)
        logger.info("upload log {} --> {}".format(local_file, hadoop_des_filename))

    def start(self):
        self.data_handler.run()
        cur_time = time.strftime("%Y%m%d%H%M%S")
        logger.info("use cur_time {} to backup database and update logs".format(cur_time))
        self.backup_database(cur_time)
        self.update_history_logs(cur_time)
