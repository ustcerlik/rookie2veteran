import numpy as np


class Nms(object):

    def __init__(self, threshold, topk):
        super(Nms, self).__init__()
        self.threshold = threshold
        self.topk = topk

    def nms_core(self, b_boxes, scores):
        """
        b_boxes 排序， 按照scores, mapping: index的映射
        注意：nms其实是对每个类别都要做，即类别之前是独立的，就是说如果有两个框即使iou很高，但是不是一个类，也不会过滤掉
        :param b_boxes:
        :param scores:
        :return:
        """
        des_indices = np.argsort(-scores)  # descending indices

        mapping = dict()
        for i in range(len(des_indices)):
            mapping[i] = des_indices[i]

        descend_boxes = [b_boxes[index] for index in des_indices]
        remove_flag = np.zeros_like(des_indices)

        for index, box in enumerate(descend_boxes):
            if remove_flag[index]:
                continue
            else:
                for i in range(index + 1, len(descend_boxes)):
                    cur_iou = self.iou(box, descend_boxes[i])
                    if cur_iou > self.threshold:
                        remove_flag[i] = 1

        indices = [mapping[index] for index, flag in enumerate(remove_flag) if not flag]
        if len(indices) <= topk:
            return indices
        return indices[0: self.topk]

    def iou(self, b_box1, b_box2):
        x1, y1, w1, h1 = b_box1
        x2, y2, w2, h2 = b_box2
        s_sum = w1 * h1 + w2 * h2

        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)

        if left >= right or top >= bottom:
            return 0
        intersect = (right - left) * (bottom - top)

        return intersect / (s_sum - intersect)


def plot_bbox(b_boxes, indices, scores):
    import matplotlib.pyplot as plt
    import cv2
    img_orig = np.zeros((int(300), int(300), 3), np.uint8)

    for index, box in enumerate(b_boxes):
        cv2.rectangle(img_orig, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]),
                      color=(255,255,255), thickness=1)
        cv2.putText(img_orig, str(scores[index]), (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1
                    )
    plt.subplot(121)
    plt.title("inference boxes")
    plt.imshow(img_orig)
    img_nms = np.zeros((int(300), int(300), 3), np.uint8)
    for index in indices:
        box = b_boxes[index]
        cv2.rectangle(img_nms, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(255, 0, 0), thickness=2)
        cv2.putText(img_nms, str(scores[index]), (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1
                    )
    plt.subplot(122)
    plt.title("after nms")
    plt.imshow(img_nms)

    plt.show()


def loop_nms(b_boxes, scores, threshold, topk):

    nms_op = Nms(threshold, topk)
    indices = nms_op.nms_core(b_boxes, scores)  # topk indices

    plot_bbox(b_boxes, indices, scores)


if __name__ == '__main__':
    # gt_boxes = np.array([[10,10,101,99], [81,79,49,51]])
    b_boxes = np.array([[40,40,100,100],[50,60,105,105],[210, 200, 65,75], [220,215,55,60],[200,30,50,50]])  # xywh
    scores = np.array([0.91,0.8,0.9, 0.7,0.6])
    threshold = 0.5
    topk = 3

    loop_nms(b_boxes, scores, threshold, topk)
