from dtw import dtw
import pickle
import time
import statistics
from numpy import zeros


times = []
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

    for p1, p2 in zip(frame1, frame2):
        dist += (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    return dist


def fun(gesture,sequence):
    global times
    gest_len = len(gesture)


    for i in range(gest_len,len(sequence)):
        t = time.time()
        d, cost_matrix, acc_cost_matrix,path = dtw(gesture, sequence[i - gest_len:i], dist=frame_distance)
        ct = time.time() - t
        print("dtw computation time: " + str(ct) + "s")
        print()
        print(cost_matrix)
        print()
        times.append(ct)
        i += 1


def acc_cost(cost, n):
    acc_cost_m = zeros((n,n)).tolist()

    acc_cost_m[0][0]=cost[0][0]

    for i in range(1,n):
        acc_cost_m[i][0] = acc_cost_m[i-1][0] + cost[i][0]

    for i in range(1,n):
        acc_cost_m[0][i] = acc_cost_m[0][i-1] + cost[0][i]

    for i in range(1,n):
        for j in range(1,n):
            acc_cost_m[i][j] = min(acc_cost_m[i-1][j],acc_cost_m[i][j-1],acc_cost_m[i-1][j-1]) + cost[i][j]

    return acc_cost_m

def get_cost(x,y,n):
    cost_m = zeros((n,n)).tolist()

    for i in range(n):
        for j in range(n):
            cost_m[i][j] = frame_distance(x[i],y[j])


    return cost_m


def fun2(gesture,sequence):
    gest_len = len(gesture)

    t = time.time()
    d, cost_matrix1, acc_cost_matrix1,path = dtw(gesture, sequence[0:gest_len], dist=frame_distance)
    ct = time.time() - t
    print("dtw computation time: " + str(ct) + "s")
    # print()
    # print(acc_cost_matrix1)
    # print()

    t = time.time()
    d, cost_matrix2, acc_cost_matrix2,path = dtw(gesture, sequence[1:gest_len+1], dist=frame_distance)
    old_ct = time.time() - t

    print(path.tolist())
    print("dtw computation time: " + str(ct) + "s")
    # print()
    # print(cost_matrix2)
    # print()

    t = time.time()
    cost_matrix1 = cost_matrix1.tolist()
    cost_matrix2 = cost_matrix2.tolist()
    acc_cost_matrix2 = acc_cost_matrix2.tolist()
    
    convesion = time.time() - t

    print("conversion time: " + str(convesion))

    print()

    t = time.time()
    naive_cost = get_cost(gesture,sequence[1:gest_len+1], gest_len)
    n_cost_t = time.time() - t
    print("naive cost time " + str(n_cost_t))

    print(naive_cost == cost_matrix2)

    t = time.time()
    custom_cost = [ cost_matrix1[i][1:] + [frame_distance(x,sequence[gest_len])] for (i,x) in enumerate(gesture)]
    cost_t = time.time() - t
    print("custom cost time: " + str(cost_t) + "s")

    t = time.time()
    my_acc_cost = acc_cost(custom_cost, gest_len)
    acc_cost_t = time.time() - t
    print("custom acc_cost time = " + str(acc_cost_t))

    # my_acc_cost  = my_acc_cost.tolist()
    print(my_acc_cost == acc_cost_matrix2)

    # print(my_acc_cost)
    # print(acc_cost_matrix2)


    new_ct = acc_cost_t + cost_t
    print("total custom time: " + str(new_ct))

    print("better than dtw by " + str(old_ct/new_ct))
    print("better than ccc by " + str(n_cost_t/cost_t))




if __name__ == "__main__":
    gesture = get_frames("/home/vladyslav/gr/tf-pose-estimation/left_forward")
    sequence = get_frames("/home/vladyslav/gr/tf-pose-estimation/gr2")

    fun2(gesture,sequence)
    #print(statistics.mean(times))
    