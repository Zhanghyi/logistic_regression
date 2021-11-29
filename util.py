import os
import json
import matplotlib.pyplot as plt

abspath = os.path.dirname(os.path.abspath(__file__))


def load_data(data_category):
    p = abspath + '/data/' + data_category + 'set.json'
    with open(p) as f:
        data = json.load(f)
    if data_category != 'test':
        return [elem[0] for elem in data], [elem[1] for elem in data]
    return [elem for elem in data]


def save_data(X, y, file_name):
    p = abspath + '/output/' + file_name
    with open(p, 'w') as f:
        f.write(json.dumps([(X[i], int(y[i])) for i in range(len(X))]))
        f.close()


def accuracy(y_pred, y_label):
    return float(sum(y_pred == y_label)) / float(len(y_label))


def draw_loss_history(loss_history):
    plt.title("loss history")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.plot(range(len(loss_history)), loss_history)
    plt.savefig(abspath + '/output/loss history.png')
