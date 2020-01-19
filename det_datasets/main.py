import logging
import os

from handler import Handler


def setup_logger(log_file):
    base_dir = os.path.dirname(log_file)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_file, mode="w")
    sh.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)
    formatters = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(filename)s [%(lineno)d] %(message)s")
    sh.setFormatter(formatters)
    fh.setFormatter(formatters)
    root_logger.addHandler(sh)
    root_logger.addHandler(fh)
    return root_logger


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cf", help="config file", required=True)
    parser.add_argument("--logfile", "-log", help="log dir", default="logs/log.log")
    parser.add_argument("--num_process", "-p", help="multi process num", default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger = setup_logger(args.logfile)
    logger.info("log file {} created!".format(args.logfile))
    handler = Handler(args.config_file, args.num_process)
    logger.info("handler created! start processing...")
    handler.start()
    logger.info("done.")
