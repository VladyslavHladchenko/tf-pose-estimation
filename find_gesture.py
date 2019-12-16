import argparse
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from tf_pose.common import CocoPart
from dtw import dtw
import operator

import numpy

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'ro')


def get_frames(filename, proc=False):
    frames = []
    with open(filename, "rb") as file:
        while True:
            try:
                if proc:
                    frames.append(process(pickle.load(file)))
                else:
                    frames.append(pickle.load(file))
            except EOFError:
                break

    print("Read " + str(len(frames)) + " frames from file '" + filename + "'")
    return frames


def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return ln,


def update(frame):
    x, y = zip(*frame)
    ln.set_data(x, y)

    print("frame " + str(update.counter))
    update.counter += 1
    return ln,

update.counter = 0


def draw_human(frames):
    FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False, interval=500)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gesture recognition with tf-pose-estimation and dtw')

    parser.add_argument('--g', type=str, default="")
    parser.add_argument('--s', type=str, default="")
    parser.add_argument('--process', type=bool, default=False)
    parser.add_argument('--p1', type=int, default=0)
    parser.add_argument('--p2', type=int, default=0)

    args = parser.parse_args()

    frames1 = get_frames(args.g, args.process)
    frames2 = get_frames(args.s, args.process)

    # p1 = args.p1
    # p2 = args.p2

    # if args.file2 != "":
    #     with open(args.file2, "wb") as file:
    #         for frame in frames1[p1:p2]:
    #             pickle.dump(frame, file)

    # draw_human(frames1)
    # draw_human(frames2)

    if len(frames1) > len(frames2):
        print("len of g > len of s")
    else:
        compare(frames1, frames2)
