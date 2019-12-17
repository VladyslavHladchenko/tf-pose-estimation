import argparse
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from tf_pose.common import CocoPart
from dtw import dtw
import operator
from termcolor import colored

import numpy

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'ro')

gestures = []
sequence = []
gesture_names = []

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


def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return ln,


def update(frame):
    x, y = zip(*frame)
    ln.set_data(x, y)

    print("frame " + str(update.counter))
    t = time.time()
    ds = []
    min_gn = ""
    min_d = 100
    for g_i, g in enumerate(gestures):
        start = len(g)

        if update.counter >= start:
            d, cost_matrix, acc_cost_matrix, path = dtw(g, sequence[update.counter - start:update.counter], dist=frame_distance)
            ds.append((gesture_names[g_i], d))
            if min_d > d:
                min_d = d
                min_gn = gesture_names[g_i]

    for gn, d in ds:
        if min_gn == gn:
            print(colored("gesture '" + gn + "' distance: " + str(d), "green"))
        else:
            print("gesture '" + gn + "' distance: " + str(d))

    # print("dtw computation time: " + str(time.time() - t) + "s")
    print()
    update.counter += 1
    return ln,

update.counter = 0


def draw_human(frames, interval):
    FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False, interval=interval)
    plt.grid()
    plt.show()


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

    if frame[1][0] != 0 and frame[4][0] != 0:
        coeff = 0.3 / abs(frame[1][0] - frame[4][0])

    frame = list(map(lambda f:  tuple(map(operator.sub, f, shift)), frame))

    frame = list(map(lambda f: tuple(map(operator.mul, f, (coeff, coeff))), frame))

    return frame


def compare(frames1, frames2):

    # TODO optimize
    def frame_distance(frame1, frame2):
        dist = 0
        for p1, p2 in zip(frame1, frame2):
            dist += numpy.linalg.norm(tuple(map(operator.sub, p1, p2)))
        return dist

    t = time.time()
    min_d = 100
    min_i = 0
    start = len(frames1)

    for i in range(start, len(frames2)):
        d, cost_matrix, acc_cost_matrix, path = dtw(frames1, frames2[i-start:i], dist=frame_distance)
        if d < min_d:
            min_d = d
            min_i = i
        print("distance: " + str(d))

    print()
    print("dtw computation time: " + str(time.time()-t))
    print("minimal distance is: " + str(min_d))
    print("start frame: " + str(min_i-start))
    draw_human(frames2[min_i-start:min_i])
    # plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
    # plt.plot(path[0], path[1], 'w')
    # plt.show()


def recognize_and_classify(sequence, gestures):
    # for g in gestures:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gesture recognition with tf-pose-estimation and dtw')

    parser.add_argument('--sequence', type=str, default="")
    parser.add_argument('--gestures', type=str, default="")
    parser.add_argument('--interval', type=int, default=250)

    args = parser.parse_args()

    sequence = get_frames(args.sequence)

    gesture_names = args.gestures.split()
    gestures = [get_frames(g) for g in gesture_names]

    # recognize_and_classify(sequence, gestures)
    draw_human(sequence,args.interval)
