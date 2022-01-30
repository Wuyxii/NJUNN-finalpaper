import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import sys

root = sys.argv[1]
size = [6, 8, 10, 12]
data_num = [64, 256, 1024, 4096]


# construct model
class model(nn.Module):
    def __init__(self, zr_dimension, d_model=16):
        super(model, self).__init__()
        self.seq_embedding = nn.Embedding(3, d_model)
        self.zr_embedding = nn.Embedding(zr_dimension, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, seq, zr):
        embedded_seq = self.seq_embedding(seq)
        embedded_zr = self.zr_embedding(zr).squeeze(1)
        h = self.transformer_encoder(embedded_seq.permute(1, 0, 2)).permute(1, 0, 2)
        h = torch.mean(h, dim=1)
        y = self.fc(torch.mul(h, embedded_zr))
        return y


# train
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        display.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.clear_output(wait=True)


def train_plot_fu(model_f, model_u, train_iter, num_epochs, lr=1e-3, device=try_gpu()):
    print('training on', device)
    model_f.to(device)
    model_u.to(device)
    loss = nn.MSELoss()

    optimizer_f = torch.optim.Adam(model_f.parameters(), lr=lr)
    optimizer_u = torch.optim.Adam(model_u.parameters(), lr=lr)

    animator = Animator(xlabel='epoch', xlim=[0, num_epochs],
                        legend=['f train loss', 'u train loss'])

    for epoch in range(num_epochs):
        l_f_list = []
        l_u_list = []
        for i, (seq, z, f, u) in enumerate(train_iter):
            model_f.train()
            model_u.train()
            seq, z, train_f, train_u = seq.to(device), z.to(device), f.to(device), u.to(device)

            f_hat = model_f(seq, z)
            l_f = loss(f_hat, train_f)
            optimizer_f.zero_grad()
            l_f.backward()
            optimizer_f.step()

            u_hat = model_u(seq, z)
            l_u = loss(u_hat, train_u)
            optimizer_u.zero_grad()
            l_u.backward()
            optimizer_u.step()

            l_f = l_f.cpu().detach().numpy()
            l_u = l_u.cpu().detach().numpy()
            l_f_list.append(l_f)
            l_u_list.append(l_u)
            if (i + 1) % 100 == 0:
                animator.add(epoch + i / len(train_iter),
                             (np.mean(l_f_list), None))
                animator.add(epoch + i / len(train_iter),
                             (None, np.mean(l_u_list)))
                l_f_list = []
                l_u_list = []
        print("Epoch: {}/{}, train f loss: {}, train u loss: {}".format(epoch, num_epochs, l_f, l_u))
        torch.save(model_f.state_dict(), "f_model.param")
        torch.save(model_u.state_dict(), "u_model.param")


def train_plot_structure(model, train_iter, num_epochs, lr=1e-3, device=try_gpu()):
    print('training on', device)
    model.to(device)
    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    animator = Animator(xlabel='epoch', xlim=[0, num_epochs],
                        legend=['structure train loss'])

    for epoch in range(num_epochs):
        l_list = []
        for i, (seq, r, structure) in enumerate(train_iter):
            model.train()
            seq, r, structure = seq.to(device), r.to(device), structure.to(device)

            structure_hat = model(seq, r)
            l = loss(structure_hat, structure)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            l = l.cpu().detach().numpy()
            l_list.append(l)
            if (i + 1) % 100 == 0:
                animator.add(epoch + i / len(train_iter),
                             (np.mean(l_list)))
                l_list = []
        print("Epoch: {}/{}, train structure loss: {}".format(epoch, num_epochs, l))
        torch.save(model.state_dict(), "structure_model.param")


# load FU data
seq = []
f_data = []
u_data = []

for i in range(4):
    pre = root + "/" + str(size[i])
    seq_filename = pre + "/sq.txt"
    with open(seq_filename, "r") as f:
        line = f.readline().split(" ")[:-1]
        line = [eval(i) for i in line]
        while line:
            line.extend([2] * (20 - len(line)))
            for t in range(48):
                seq.append(line)
            line = f.readline().split(" ")[:-1]
            line = [eval(i) for i in line]
    for j in range(1, data_num[i] + 1):
        energy_filename = pre + "/FU_" + str(j) + ".txt"
        with open(energy_filename, "r") as f:
            line = f.readline().strip("\n").split(" ")
            line = [eval(i) for i in line]
            while line:
                f_data.append([line[0]])
                u_data.append([line[1]])
                line = f.readline().strip("\n").split(" ")
                if line == ['']:
                    break
                line = [eval(i) for i in line]


seq = torch.LongTensor(seq)
z = torch.LongTensor([[float(i)] for i in range(0, 48)] * int(len(seq) / 48))
f = torch.tensor(f_data)
u = torch.tensor(u_data)

dataset_fu = TensorDataset(seq, z, f, u)
train_loader_fu = DataLoader(dataset=dataset_fu, batch_size=128, shuffle=True)

# train FU
F_model = model(48)
U_model = model(48)
Structure_model = model(480)
train_plot_fu(F_model, U_model, train_loader_fu, 10)
plt.show()

# load structure data
seq = []
structure = []
r = []
for i in range(4):
    pre = root + "/" + str(size[i])
    seq_filename = pre + "/sq.txt"
    seq_temp = []
    with open(seq_filename, "r") as f:
        line = f.readline().split(" ")[:-1]
        line = [eval(i) for i in line]
        while line:
            line.extend([2] * (20 - len(line)))
            seq_temp.append(line)
            line = f.readline().split(" ")[:-1]
            line = [eval(i) for i in line]
    for j in range(1, data_num[i] + 1):
        structure_filename = pre + "/Structure_" + str(j) + ".txt"
        with open(structure_filename, "r") as f:
            line = f.readline().strip("\n").split(" ")
            line = [eval(i) for i in line]
            r_dis = 0
            while line:
                temp = float(np.mean(line))
                if temp != 0:
                    structure.append([temp])
                    r.append([r_dis])
                    seq.append(seq_temp[j-1])
                line = f.readline().strip("\n").split(" ")
                if line == ['']:
                    break
                line = [eval(i) for i in line]
                r_dis = r_dis + 1

seq = torch.LongTensor(seq)
r = torch.LongTensor(r)
structure = torch.tensor(structure)
dataset_structure = TensorDataset(seq, r, structure)
train_loader_structure = DataLoader(dataset=dataset_structure, batch_size=128, shuffle=True)

# train structure
structure_model = model(480)
train_plot_structure(structure_model, train_loader_structure, 10)
plt.show()
