import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


class VideoHandler(object):
    def __init__(self, args):
        super(VideoHandler, self).__init__()
        self.args = args
        self.check_args()
        self.func = self.img2video if args.mode == "img2video" else self.video2img

    @staticmethod
    def get_frame_indices(start, end, step, frame_need):
        if frame_need != -1:
            assert not frame_need > (end - start)
            indices = np.linspace(start, end, frame_need).tolist()
        else:
            indices = np.linspace(start, end, int((end - start) / step)).tolist()

        indices = [int(index + 0.5) for index in indices]
        return indices

    def img2video(self):
        # do not need check args
        fps = self.args.fps
        img_path = self.args.img_path
        imgs = [img for img in os.listdir(img_path) if (".jpg" in img)]
        imgs.sort()
        print(len(imgs))
        print(imgs[0])
        img_size = cv2.imread(os.path.join(img_path, imgs[0])).shape[0:2]
        des_video = self.args.des_video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(des_video, fourcc, fps, (img_size[1], img_size[0]))
        for img in tqdm(imgs):
            frame = cv2.imread(os.path.join(img_path, img))
            vw.write(frame)

        vw.release()

    def video2img(self):
        # handle start end frame_need
        video_path = self.args.video_path
        video_name = video_path.split("/")[-1]
        cap = cv2.VideoCapture(video_path)
        num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        start = 0 if self.args.start == -1 else self.args.start
        end = num_frame - 1 if self.args.end == -1 or self.args.end >= num_frame - 1 else self.args.end
        step = self.args.step
        frame_need = self.args.frame_need
        frames = set(self.get_frame_indices(start, end, step, frame_need))
        max_index = max(frames)
        min_index = min(frames)
        print("total ext frames : {}, from {} frames".format(len(frames), (max_index - min_index)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_index)  # set waste lots of time , do not use it frequently

        for cur_index in tqdm(range(min_index, max_index)):
            if cur_index in frames:
                # cap.grab()
                ret, img = cap.retrieve()
                file_name = "%05d" % cur_index
                cv2.imwrite(os.path.join(self.args.des_folder, video_name, "{}.jpg".format(file_name)), img) if ret \
                    else print("save {} error".format(file_name))

            cap.grab()

        cap.release()

    def run(self):
        self.func()

    def check_args(self):
        args = self.args
        if args.mode == "video2img":
            assert os.path.isfile(args.video_path), "{} is not file".format(args.video_path)
            video_name = args.video_path.split("/")[-1]

            if os.path.exists(os.path.join(args.des_folder, video_name)):
                shutil.rmtree(os.path.join(args.des_folder, video_name))
                # assert not os.listdir(args.des_folder), "{} not empty".format(args.des_folder)
            os.makedirs(os.path.join(args.des_folder, video_name))

            assert args.start
            assert args.end
            assert args.frame_need
            assert args.step

        else:
            assert os.path.isdir(args.img_path), "{} is not a dir".format(args.img_path)
            assert os.listdir(args.img_path), "{} is empty".format(args.img_path)
            assert not os.path.isfile(args.des_video), "{} exists".format(args.des_video)
            if not os.path.exists(os.path.dirname(args.des_video)):
                os.makedirs(os.path.dirname(args.des_video))
            assert args.fps, "fps error"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="img2video  |  video2img", choices=["video2img", "img2video"], required=True)

    # for video 2 img
    parser.add_argument("--video_path", "-v", help="video path for video2img")
    parser.add_argument("--des_folder", "-df", help="des folder for video2img")
    parser.add_argument("--start", "-s", help="frame start", default=-1, type=int)
    parser.add_argument("--end", "-e", help="frame end", default=-1, type=int)
    parser.add_argument("--step", "-step", help="step", default=1, type=int)
    parser.add_argument("--frame_need", "-fn", help="total images need", default=-1, type=int)

    # for img 2 video
    parser.add_argument("--img_path", "-i", help="images path for img2video")
    parser.add_argument("--des_video", "-dv", help="des video path for img2video", default="./des.mp4")
    parser.add_argument("--fps", help="fps for img2video", default=12, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # TODO add multiprocessing if video_path is a list. Now only a single file name support.
    vh = VideoHandler(args)
    vh.run()
    print("done")


if __name__ == '__main__':
    main()
