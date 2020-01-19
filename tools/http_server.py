import argparse
import os
import sys

"""
usage: python http_server.py 10086
"""

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    args, port = argparser.parse_known_args()
    version = sys.version

    if version.startswith('2'):
        os.system("python -m SimpleHTTPServer {}".format(port[0]))
    else:
        os.system("python -m http.server {}".format(port[0]))
