import os
import shutil
from functools import partial

from tqdm import tqdm


class FileOp(object):

    def __init__(self):
        super(FileOp, self).__init__()

    def exists(self, file_name):
        if os.path.isfile(file_name):
            return False
        return True

    def copy_file(self, img_name, des_path, base_path, prefix=None):
        cur_path = os.path.join(base_path, img_name)
        assert self.exists(cur_path), "file not exists"
        des_name = prefix + "_" + img_name if prefix else img_name
        new_des_path = os.path.join(des_path, des_name)
        shutil.copy(cur_path, new_des_path)

    def move_file(self, img_name, des_path, base_path, prefix=None):
        cur_path = os.path.join(base_path, img_name)
        assert self.exists(cur_path), "file not exists"
        des_name = prefix + "_" + img_name if prefix else img_name
        cur_des = os.path.join(des_path, des_name)
        if os.path.exists(cur_des):
            os.remove(cur_des)
        shutil.move(cur_path, cur_des)

    def check_size(self, img_name, base_path):
        cur_path = os.path.join(base_path, img_name)
        assert self.exists(cur_path), "file not exists"
        file_size = os.path.getsize(cur_path)
        return file_size

    def del_file(self, img_name, base_path):
        cur_path = os.path.isfile(base_path, img_name)
        assert self.exists(cur_path), "file not exists"
        os.remove(cur_path)
        # print("{} moved.".format(img_name))

    def build_tars(self, file_list, target_name, mode, arcname="./{}"):
        """
        do not use multiprocess
        :param file_list: it should be absolute path
        :param target_name:
        :param arcname:  like "./train/{}" , the tree of tarfile
        :param mode: a w
        :return:
        """
        import tarfile
        target_tar = tarfile.open(target_name, mode)
        for cur_file in tqdm(file_list):
            cur_file_name = cur_file.split("/")[-1]
            target_tar.add(cur_file, arcname=arcname.format(cur_file_name))

        target_tar.close()
        print("done!")

    def test(self, i, ii, iii):
        print("tf2_start", i)
        print("tf2_start", ii)
        print("tf2_start", iii)


class Mp(object):

    def __init__(self, process_count=None):
        super(Mp, self).__init__()
        from multiprocessing import Pool
        import multiprocessing
        self.process_count = process_count if process_count else multiprocessing.cpu_count()
        self.pool = Pool(self.process_count)

    def mapping(self, func, mapping_input):
        self.pool.map(func, mapping_input)


def partial_fun(func, *args):
    """
    还是别用这个了，默认的挺好用的
    the first args can not in kwargs, its should be the input args of partial function
    args must be sequential, like func input
    :param func: function name
    :param args: partial args
    :return: partial function
    """
    from functools import partial
    return partial(func, *args)


if __name__ == '__main__':

    fo = FileOp()
    mp = Mp()
    input_data = [1,2,3]
    # a = partial_fun(fo.tf2_start, 2, 3)
    partial_fun = partial(fo.test, ii="/ssd/kli", iii="dd")
    mp.mapping(partial_fun, input_data)
    print("done!")
