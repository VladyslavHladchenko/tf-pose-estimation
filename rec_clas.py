import time
import pickle
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from tf_pose.common import CocoPart
from dtw import dtw
import operator
from termcolor import colored
import logging
# import shutil
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from CocoKeyPoints import CocoPart as keyPoints

import cv2
# import numpy as np

# import numpy



# fig, ax = plt.subplots()
# ln, = plt.plot([], [], 'ro')

# gestures = []
# sequence = []
# gesture_names = []

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

frames = []
gesture_names = []
gestures = []
gesture_thresholds = {}
max_gesture_len = 0

ESC_KEY=27

# gn_threshold = {"left_side_2":0.2, "left_forward":0.3, "right_side_2":0.3, "right_forward_2":0.2, "left_circle_2":0.6, "right_circle_2":0.6}

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
    # prev_cost=[]
    #TODO zip
    for g_i, g in enumerate(gestures):
        start = len(g)
        # t = time.time()
        d, cost_matrix, acc_cost_matrix, path = dtw(g, frames[max_gesture_len - start:max_gesture_len], dist=frame_distance)
        # ct = time.time() - t
        # print("dtw computation time for " + gesture_names[g_i] + ":" + str(ct) + "s")
        # print()
        # print(cost_matrix)
        # print()
        ds.append((gesture_names[g_i], d))
        if min_d > d:
            min_d = d
            min_gn = gesture_names[g_i]

        # prev_cost = cost_matrix
    for gn, d in ds:
        if min_gn == gn and min_d < gesture_thresholds[gn]:
            print(colored("gesture '" + gn + "' distance: " + str(d), "green"))
        else:
            pass
            # print("gesture '" + gn + "' distance: " + str(d))

    # print()
    frames = frames[1:max_gesture_len]



def get_frames(filename):
    frames = []
    with open(filename, "rb") as file:
        while True:
            try:
                frames.append(pickle.load(file))
            except EOFError:
                break

    print("Read " + str(len(frames)) + " frames from file '" + filename + "'")
    return frames


def frame_distance(frame1, frame2):
    dist = 0
    # print("frame_distance")
    # print(frame1)
    # print(frame2)

    for p1, p2 in zip(frame1, frame2):
        dist += (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    return dist


# def init():
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     return ln,


# def update(frame):
#     x, y = zip(*frame)
#     ln.set_data(x, y)

#     print("frame " + str(update.counter))
#     t = time.time()
#     ds = []
#     min_gn = ""
#     min_d = 100
#     for g_i, g in enumerate(gestures):
#         start = len(g)

#         if update.counter >= start:
#             d, cost_matrix, acc_cost_matrix, path = dtw(g, sequence[update.counter - start:update.counter], dist=frame_distance)
#             ds.append((gesture_names[g_i], d))
#             if min_d > d:
#                 min_d = d
#                 min_gn = gesture_names[g_i]

#     for gn, d in ds:
#         if min_gn == gn:
#             print(colored("gesture '" + gn + "' distance: " + str(d), "green"))
#         else:
#             print("gesture '" + gn + "' distance: " + str(d))

#     # print("dtw computation time: " + str(time.time() - t) + "s")
#     print()
#     update.counter += 1
#     return ln,

# update.counter = 0


# def draw_human(frames, interval):
#     FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False, interval=interval)
#     plt.grid()
#     plt.show()


def process(frame):
    def invert_y(t):
        return t[0], 1-t[1]

    frame = frame[0:7]  # one human

    # for index, point in enumerate(frame):
    #     s = str(index)
    #     if point[0] == 0 or point[1] == 0:
    #         s = "zero at frame " + str(index) + ": " + str(CocoPart(index + 1)) + ": " + str(point)
    #         print(s)


    frame = list(map(invert_y, frame))

    # return frame

    shift = frame[0]

    coeff = 1

    if frame[1][0] != 0 and frame[4][0] != 0 and frame[1][0] != frame[4][0]:
        coeff = 0.3 / abs(frame[1][0] - frame[4][0])

    frame = list(map(lambda f:  tuple(map(operator.sub, f, shift)), frame))

    frame = list(map(lambda f: tuple(map(operator.mul, f, (coeff, coeff))), frame))

    return frame


# def compare(frames1, frames2):

#     # TODO optimize
#     def frame_distance(frame1, frame2):
#         dist = 0
#         for p1, p2 in zip(frame1, frame2):
#             dist += numpy.linalg.norm(tuple(map(operator.sub, p1, p2)))
#         return dist

#     t = time.time()
#     min_d = 100
#     min_i = 0
#     start = len(frames1)

#     for i in range(start, len(frames2)):
#         d, cost_matrix, acc_cost_matrix, path = dtw(frames1, frames2[i-start:i], dist=frame_distance)
#         print()
#         print(cost_matrix)
#         print()
#         if d < min_d:
#             min_d = d
#             min_i = i
#         print("distance: " + str(d))

#     print()
#     print("dtw computation time: " + str(time.time()-t))
#     print("minimal distance is: " + str(min_d))
#     print("start frame: " + str(min_i-start))
#     draw_human(frames2[min_i-start:min_i])
#     # plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
#     # plt.plot(path[0], path[1], 'w')
#     # plt.show()


def load_gestures():
    global max_gesture_len
    global gesture_names
    global gesture_thresholds
    global gestures

    with open('gestures') as file:
        #TODO catch file not found? 
        for line in file:
            if line.startswith('#') or line == "" or line.count(' ') != 1:
                continue
            name, threshold = line.split()
            
            gesture_names.append(name)
            gesture_thresholds[name] = float(threshold)

    gestures = [get_frames(name) for g in gesture_names]

    #TODO smarter
    for g in gestures:
        if len(g) > max_gesture_len:
            max_gesture_len = len(g)


def run(camera, resize):
    global frames
    model='mobilenet_thin'
    fps_time = 0

    load_gestures()

    #TODO if gestues are empty

    # logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
    w, h = model_wh(resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h), trt_bool=False)
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368), trt_bool=False)
    cam = cv2.VideoCapture(camera)

    # ret_val, image = cam.read()
    # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    while True:
        ret_val, image = cam.read()
        human = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        coords = []
        if human:
            for i in range(1,8):
                if i not in human.body_parts.keys():
                    coords.append((0, 0))
                    continue
                body_part = human.body_parts[i]
                coords.append((body_part.x, body_part.y))

            frames.append(process(coords))
            detect_and_classify()

        image = TfPoseEstimator.draw_humans(image, human, imgcopy=False)

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == ESC_KEY:
            break

    cv2.destroyAllWindows()
