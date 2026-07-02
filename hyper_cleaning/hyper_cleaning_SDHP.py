import torch
import random
from torchvision.datasets import MNIST, FashionMNIST
import torch.nn.functional as F
import copy
import numpy as np
import time
import csv
import argparse
import math

parser = argparse.ArgumentParser(description='Data HyperCleaner')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='FashionMNIST', metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--y_loop', type=int, default=1)
parser.add_argument('--x_loop', type=int, default=500)
parser.add_argument('--x_lr', type=float, default=0.5)
parser.add_argument('--y_lr', type=float, default=0.5)
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--convex', action='store_true', default=False)

parser.add_argument('--rho', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=1.5)
args = parser.parse_args()
METHOD_NAME = 'SDHP'
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset == 'MNIST':
    dataset = MNIST(root=r"./data/mnist", train=True, download=True)
elif args.dataset == 'FashionMNIST':
    dataset = FashionMNIST(root=r"./data/fashionmnist", train=True, download=True)
print(args)

def update_tensor(hparams, detas, step):
    for p, d in zip(hparams, detas):
        if d is not None:
            p.data += step * d
            # p.data -= step * d

class Dataset:
    def __init__(self, data, target, polluted=False, rho=0.0):
        self.data = data.float() / torch.max(data)
        print(list(target.shape))
        if not polluted:
            self.clean_target = target
            self.dirty_target = None
            self.clean = np.ones(list(target.shape)[0])
        else:
            self.clean_target = None
            self.dirty_target = target
            self.clean = np.zeros(list(target.shape)[0])
        self.polluted = polluted
        self.rho = rho
        self.set = set(target.numpy().tolist())

    def data_polluting(self, rho):
        assert self.polluted == False and self.dirty_target is None
        number = self.data.shape[0]
        number_list = list(range(number))
        random.shuffle(number_list)
        self.dirty_target = copy.deepcopy(self.clean_target)
        for i in number_list[:int(rho * number)]:
            dirty_set = copy.deepcopy(self.set)
            dirty_set.remove(int(self.clean_target[i]))
            self.dirty_target[i] = random.randint(0, len(dirty_set))
            self.clean[i] = 0
        self.polluted = True
        self.rho = rho

    def data_flatten(self):
        try:
            self.data = self.data.view(self.data.shape[0], self.data.shape[1] * self.data.shape[2])
        except BaseException:
            self.data = self.data.reshape(self.data.shape[0],
                                          self.data.shape[1] * self.data.shape[2] * self.data.shape[3])

    # def get_batch(self,batch_size):

    def to_cuda(self):
        self.data = self.data.cuda()
        if self.clean_target is not None:
            self.clean_target = self.clean_target.cuda()
        if self.dirty_target is not None:
            self.dirty_target = self.dirty_target.cuda()


def data_splitting(dataset, tr, val, test):
    assert tr + val + test <= 1.0 or tr > 1

    number = dataset.targets.shape[0]
    number_list = list(range(number))
    random.shuffle(number_list)
    if tr < 1:
        tr_number = tr * number
        val_number = val * number
        test_number = test * number
    else:
        tr_number = tr
        val_number = val
        test_number = test

    train_data = Dataset(dataset.data[number_list[:int(tr_number)], :, :],
                         dataset.targets[number_list[:int(tr_number)]])
    val_data = Dataset(dataset.data[number_list[int(tr_number):int(tr_number + val_number)], :, :],
                       dataset.targets[number_list[int(tr_number):int(tr_number + val_number)]])
    test_data = Dataset(
        dataset.data[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)], :, :],
        dataset.targets[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)]])
    return train_data, val_data, test_data


def accuary(out, target):
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
    return acc


def Binarization(x):
    x_bi = np.zeros_like(x)
    for i in range(x.shape[0]):
        # print(x[i])
        x_bi[i] = 1 if x[i] >= 0 else 0
    return x_bi


class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(tr.data.shape[0]).to("cuda").requires_grad_(True))

    def forward(self, y):
        # if torch.norm(torch.sigmoid(self.x), 1) > 2500:
        #     y = torch.sigmoid(self.x) / torch.norm(torch.sigmoid(self.x), 1) * 2500 * y
        # else:
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y


def tonp(x):
    return x.detach().cpu().numpy()


def normlize(P, delta=1):
    device = P[0].device
    dtype = P[0].dtype

    sqnorm = torch.zeros((), device=device, dtype=dtype)
    for p in P:
        sqnorm = sqnorm + p.pow(2).sum()

    scale = 1.0 / torch.sqrt(delta * sqnorm + 1)
    P_normed = [p * scale for p in P]  # v / sqrt(δ||v||^2 + 1)
    return P_normed


tr, val, test = data_splitting(dataset, 5000, 5000, 10000)
tr.data_polluting(args.pollute_rate)
tr.data_flatten()
val.data_flatten()
test.data_flatten()
tr.to_cuda()
val.to_cuda()
test.to_cuda()
cuda = torch.cuda.is_available()
log_path = "results/{}/hclean_seed{}_yloop{}_{}_{}_xlr{}_ylr{}_rho{}_gamma{}_nonconvex_{}.csv".format(args.dataset,args.seed,args.y_loop,
                                                                                                    METHOD_NAME,
                                                                                                    args.dataset,
                                                                                                    args.x_lr,
                                                                                                    args.y_lr,
                                                                                                    args.rho,
                                                                                                    args.gamma,
                                                                                                    time.strftime(
                                                                                                        "%Y_%m_%d_%H_%M_%S"))
with open(log_path, 'a', encoding='utf-8') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([args])
    csv_writer.writerow(['', 'acc', 'F1 score', 'total_hyper_time', 'total_time', 'Val loss', 'Val acc', 'p', 'r'])
d = 28 ** 2
n = 10
y_loop = args.y_loop
x_loop = args.x_loop
x = Net_x(tr)
if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
    d = 28 ** 2
    n = 10
    if args.convex:
        y = torch.nn.Sequential(torch.nn.Linear(d, n)).to("cuda")
    # non-convex
    else:
        y = torch.nn.Sequential(torch.nn.Linear(d, 250), torch.nn.ReLU(), torch.nn.Linear(250, n)).to("cuda")
val_losses = []


def lf(x_temp, y_temp, args=None):
    y_pre = y_temp(tr.data)
    loss = F.cross_entropy(y_pre, tr.dirty_target, reduction='none')
    out = x_temp(loss)
    return out


def uF(x_temp, y_temp, args=None):
    val_loss = F.cross_entropy(y_temp(val.data), val.clean_target)
    val_losses.append(tonp(val_loss))
    return val_loss


x_opt = torch.optim.Adam(x.parameters(), lr=args.x_lr)
# x_opt = torch.optim.SGD(x.parameters(), lr=args.x_lr)
acc_history = []
clean_acc_history = []

loss_x_l = 0
F1_score_last = 0
lr_decay_rate = 1
reg_decay_rate = 1
dc = 0
total_time = 0
total_hyper_time = 0
v = -1

lamb = [torch.zeros_like(p) for p in y.parameters()]
Py = [torch.zeros_like(p) for p in y.parameters()]
Px = [torch.zeros_like(p) for p in x.parameters()]
exp_y = math.exp(-args.gamma * args.y_lr)
exp_x = math.exp(-args.gamma * args.x_lr)
with torch.no_grad():
    out = y(test.data)
    acc = accuary(out, test.clean_target)
    x_bi = Binarization(x.x.cpu().numpy())
    clean = x_bi * tr.clean
    acc_history.append(acc)
    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
    r = clean.mean() / (1. - tr.rho)
    F1_score = 2 * p * r / (p + r + 1e-8)
    dc = 0
    if F1_score_last > F1_score:
        dc = 1
    F1_score_last = F1_score
    valLoss = F.cross_entropy(out, test.clean_target)
    val_out = y(val.data)
    val_acc = accuary(val_out, val.clean_target)
    eta = exp_y
    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val_loss={:.3f},val_acc={:.3f},eta={:.3f},time={:.3f}'.format(
        0,
        100 * accuary(out,
                      test.clean_target),
        100 * p, 100 * r, 100 * F1_score, valLoss, 100 * val_acc, eta, total_time))
    with open(log_path, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)

        csv_writer.writerow(
            [0, 100. * acc, 100. * F1_score, total_hyper_time, total_time, valLoss.item(), 100 * val_acc, 100 * p,
             100 * r, eta])
for x_itr in range(x_loop):
    t0 = time.time()
    x_opt.zero_grad()

    t0 = time.time()
    lower_loss, upper_loss = lf(x_temp=x, y_temp=y), uF(x_temp=x, y_temp=y)
    dfdy = torch.autograd.grad(upper_loss, list(y.parameters()), create_graph=True, retain_graph=True,
                               allow_unused=True)
    dgdy = torch.autograd.grad(lower_loss, list(y.parameters()), create_graph=True, retain_graph=True,
                               allow_unused=True)
    vec = [lam + args.rho * g for lam, g in zip(lamb, dgdy)]
    dhdy = torch.autograd.grad(dgdy, list(y.parameters()), grad_outputs=vec, retain_graph=True, allow_unused=True)
    grads_x = torch.autograd.grad(dgdy, list(x.parameters()), grad_outputs=vec, allow_unused=True)

    grads_y = [g1 + g2 for g1, g2 in zip(dfdy, dhdy)]
    Py = [exp_y * p - args.y_lr * g for p, g in zip(Py, grads_y)]
    Py_normed = normlize(Py)
    update_tensor(list(y.parameters()), Py_normed, args.y_lr)
    hyper_time = time.time()
    Px = [exp_x * p - args.x_lr * g for p, g in zip(Px, grads_x)]
    Px_normed = normlize(Px)
    update_tensor(list(x.parameters()), Px_normed, args.x_lr)

    lower_loss = lf(x_temp=x, y_temp=y)
    dgdy = torch.autograd.grad(lower_loss, list(y.parameters()), allow_unused=True)
    lamb = [0.99 * lam + args.rho * g for lam, g in zip(lamb, dgdy)]
    step_time = time.time() - t0
    total_hyper_time += time.time() - hyper_time
    total_time += time.time() - t0

    if x_itr != 0 and x_itr % 10 == 0:
        with torch.no_grad():
            out = y(test.data)
            acc = accuary(out, test.clean_target)
            x_bi = Binarization(x.x.cpu().numpy())
            clean = x_bi * tr.clean
            acc_history.append(acc)
            p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
            r = clean.mean() / (1. - tr.rho)
            F1_score = 2 * p * r / (p + r + 1e-8)
            dc = 0
            if F1_score_last > F1_score:
                dc = 1
            F1_score_last = F1_score
            valLoss = F.cross_entropy(out, test.clean_target)
            val_out = y(val.data)
            val_acc = accuary(val_out, val.clean_target)
            eta = exp_y
            print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val_loss={:.3f},val_acc={:.3f},eta={:.3f},time={:.3f}'.format(
                x_itr,
                100 * accuary(out,
                              test.clean_target),
                100 * p, 100 * r, 100 * F1_score, valLoss, 100 * val_acc, eta, total_time))
            with open(log_path, 'a', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)

                csv_writer.writerow(
                    [x_itr, 100. * acc, 100. * F1_score, total_hyper_time, total_time, valLoss.item(), 100 * val_acc, 100 * p,
                     100 * r, eta])
