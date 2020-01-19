# -*- coding:utf-8 -*-
# @Author:      likuan kli@aibee.com
# @Time:        2020-01-17 14:27
# @FILE_NAME:   switch_hdfs.py
import os
import shutil
import stat


def switch_hdfs(city, config_src, config_dst):
    assert city in ["bj", "sh", "gz"]
    src_file_dir = os.path.join(config_src, "hadoop_configs/{}-config".format(city))
    dst_file_dir = os.path.join(config_dst, "etc/hadoop".format(city))
    assert os.path.exists(src_file_dir)
    assert os.path.exists(dst_file_dir)
    os.chmod(src_file_dir, stat.S_IRWXO)
    os.chmod(dst_file_dir, stat.S_IRWXO)
    shutil.copy(os.path.join(src_file_dir, "hdfs-site.xml"), dst_file_dir)
    shutil.copy(os.path.join(src_file_dir, "core-site.xml"), dst_file_dir)
    shutil.copy(os.path.join(src_file_dir, "hadoop-env.sh"), dst_file_dir)
    shutil.copy(os.path.join(src_file_dir, "log4j.properties"), dst_file_dir)
    os.environ["IDC"] = city


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", "-c", help="city", required=True)
    parser.add_argument("--config_src", "-cs", help="python_basics path", default="/workspace/tools/python_basics/")
    parser.add_argument("--config_dst", "-cd", help="hadoop-2.6.5 path ", default="/opt/package/hadoop-2.6.5")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    switch_hdfs(args.city, args.config_src, args.config_dst)
    print("done")
