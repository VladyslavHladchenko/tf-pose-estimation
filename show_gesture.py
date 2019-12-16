import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import pickle

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'ro')


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


def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return ln,


def update(frame):
    x, y = zip(*frame)
    ln.set_data(x, y)
    print("frame " + str(update.counter))
    # plt.text(1, 1, "f" + str(update.counter))
    update.counter += 1
    return ln,

update.counter = 0

def draw_frames(frames, interval):
    FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False, interval=interval)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="")
    parser.add_argument('--interval', type=int, default=500)
    args = parser.parse_args()

    frames = get_frames(args.file)
    draw_frames(frames, args.interval)
