import argparse
import logging
import time
import shutil
import rec_clas
from termcolor import colored
from dtw import dtw

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
frames = []
gesture_names = []
gestures = []
max_gesture_len = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def detect_and_classify():
    global frames

    # print(max_gesture_len)
    if len(frames) < max_gesture_len:
        # print("len(frames) < max_gesture_len")
        return

    t = time.time()
    ds = []
    min_gn = ""
    min_d = 100
    for g_i, g in enumerate(gestures):
        start = len(g)
        d, cost_matrix, acc_cost_matrix, path = dtw(g, frames[max_gesture_len - start:max_gesture_len], dist=rec_clas.frame_distance)
        ds.append((gesture_names[g_i], d))
        if min_d > d:
            min_d = d
            min_gn = gesture_names[g_i]

    for gn, d in ds:
        if min_gn == gn and min_d < 1:
            print(colored("gesture '" + gn + "' distance: " + str(d), "green"))
        else:
            print("gesture '" + gn + "' distance: " + str(d))

    print("dtw computation time: " + str(time.time() - t) + "s")
    print()
    frames = frames[1:max_gesture_len]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')

    parser.add_argument('--tensorrt', type=str, default="False",
                       help='for tensorrt process.')

    parser.add_argument('--gestures', type=str, default="")

    args = parser.parse_args()

    gesture_names = args.gestures.split()
    gestures = [rec_clas.get_frames(g) for g in gesture_names]

    for g in gestures:
        if len(g) > max_gesture_len:
            max_gesture_len = len(g)

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        human = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        # draw point
        coords = []
        if human:
            for i in range(1, 8):
                if i not in human.body_parts.keys():
                    coords.append((0, 0))
                    continue
                body_part = human.body_parts[i]
                coords.append((body_part.x, body_part.y))

            frames.append(rec_clas.process(coords))
            detect_and_classify()

        image = TfPoseEstimator.draw_humans(image, human, imgcopy=False)

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
