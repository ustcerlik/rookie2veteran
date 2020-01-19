import argparse
import json
import os

import cv2
from tqdm import tqdm

color_dict = {'blue': (255, 0, 0), 'red': (0, 0, 255), 'green': (0, 255, 0), 'cyan': (255, 255, 0),
              'yellow': (0, 255, 255), 'pink': (255, 0, 255), 'black': (0, 0, 0), 'white': (255, 255, 255),
              'orange': (0, 97, 255)}
color = list(color_dict.keys())


def check_paths(paths):
    for path in paths:
        if not os.path.isfile(path):
            return False
    return True


def build_image_id_dict(paths):
    """
    this function can deal with different type of image_id in annotations
    but its need ground truth images to mapping id --> file_name
    :param paths:
    :return:
    """
    image_id_dict = {}
    for path in paths:
        cur_data = json.load(open(path, "r"))
        if type(cur_data) is dict:
            images = cur_data["images"]

            for im in images:
                image_id_dict[im["id"]] = im["file_name"]
            return image_id_dict  # mapping

    # paths are all dts
    assert not isinstance(cur_data[0]["image_id"], int), "Can not mapping int image_id to file_name"
    for an in cur_data:
        image_id_dict[an["image_id"]] = an["image_id"]
    return image_id_dict


def build_bbox(annos, image_id_dict, c_ids, thres, img_list, res, color_base):
    img_set = set(img_list)
    for an in annos:
        imgname = image_id_dict[an["image_id"]]
        if an["category_id"] in c_ids and imgname in img_set:
            an["color"] = color_dict[color[color_base + c_ids.index(an["category_id"])]]
            if imgname in res:
                res[imgname].append(an)
            else:
                res[imgname] = [an]


def load_files(paths, c_ids, thres, img_list):
    """
    this function can deal with both gt and dt formation
    :param paths:
    :param c_ids: [1, 2, 3]
    :param thres:  [0.7, 0.4, 0.7]
    :param img_list:
    :return:
    """
    res = {}
    image_id_dict = build_image_id_dict(paths)

    for index, path in enumerate(paths):
        cur_data = json.load(open(path, "r"))
        color_base = len(c_ids) * index
        if type(cur_data) is dict:
            build_bbox(cur_data["annotations"], image_id_dict, c_ids, thres, img_list, res, color_base)
        elif type(cur_data) is list:
            build_bbox(cur_data, image_id_dict, c_ids, thres, img_list, res, color_base)

    return res


def draw_bbox(annos, img_data, thres, c_ids):
    for an in annos:
        x1, y1, w, h = an['bbox']
        x2, y2 = x1 + w, y1 + h
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        thre_index = c_ids.index(an["category_id"])
        if "score" in an and an["score"] > thres[thre_index]:
            cv2.putText(img_data, str(round(an["score"], 2)), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, an["color"], 1)
            cv2.rectangle(img_data, (x1, y1), (x2, y2), color=an["color"], thickness=2)
        elif "score" not in an:
            cv2.rectangle(img_data, (x1, y1), (x2, y2), color=an["color"], thickness=2)
        else:
            continue

    return img_data


def build_input_data(img_list, results):
    data_input = []
    for im in img_list:
        if im in results:
            data_input.append(tuple((im, results[im])))
    return data_input


def draw_save_img(input_im_res, image_folder, thres, c_ids, des_folder):
    img_name, cur_annos = input_im_res[0], input_im_res[1]
    img_data = cv2.imread(os.path.join(image_folder, img_name))
    img_data = draw_bbox(cur_annos, img_data, thres, c_ids)
    cv2.imwrite(os.path.join(des_folder, img_name), img_data)


def draw_save_img_loop(img_list, results, image_folder, thres, c_ids, des_folder):
    input_im_res = build_input_data(img_list, results)
    from multiprocessing import Pool
    from functools import partial
    pool = Pool(8)
    new_func = partial(draw_save_img, image_folder=image_folder, thres=thres,
                       c_ids=c_ids, des_folder=des_folder)
    pool.map(new_func, tqdm(input_im_res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', '-i', help='Images folder', required=True)
    parser.add_argument('--labels', '-l', help='labels to show, ex. 1,2,3', required=True)
    parser.add_argument('--labels_threshold', '-lt', help='threshold for each label, 0.9,0.7')
    parser.add_argument('--save_results', '-s', help='save results, put folder to save here')
    parser.add_argument('--downsize_ratio', '-dr', help='show image in smaller size, only for save', default=1.5)
    args, paths = parser.parse_known_args()

    assert check_paths(paths), "paths not exist"
    img_list = os.listdir(args.image_folder)
    c_ids = [int(c_id) for c_id in args.labels.split(',')]
    thres = [float(thre) for thre in args.labels_threshold.split(",")]
    assert len(c_ids) == len(thres), "labels and thres should have the same shape"
    results = load_files(paths, c_ids, thres, img_list)

    c_index = 0
    for path in paths:
        for c_id in c_ids:
            print("path: {}, c_id: {}, color: {}".format(path, c_id, color[c_index]))
            c_index += 1

    if args.save_results:
        draw_save_img_loop(img_list, results, args.image_folder,
                           thres, c_ids, args.save_results)

    else:
        print("still developing...")
