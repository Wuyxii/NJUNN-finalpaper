import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
import csv
import numpy as np
import sys

# predict
seq = pd.read_csv(sys.argv[4], header=None)
seq = torch.tensor(data=seq.values)
z = torch.LongTensor([[float(i)] for i in range(0, 48)])
r = torch.LongTensor([[float(i)] for i in range(0, 480)])


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class model(nn.Module):
    def __init__(self, zr_dimension, d_model=16):
        super(model, self).__init__()
        self.embedding = nn.Embedding(3, d_model)
        self.z_embedding = nn.Embedding(zr_dimension, d_model)
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

    def forward(self, seq, z):
        embedded_seq = self.embedding(seq)
        embedded_z = self.z_embedding(z).squeeze(1)
        h = self.transformer_encoder(embedded_seq.permute(1, 0, 2)).permute(1, 0, 2)
        h = torch.mean(h, dim=1)
        mul_res = self.fc(torch.mul(h, embedded_z))
        y = mul_res
        return y


F_model = model(48)
U_model = model(48)
F_model.load_state_dict(torch.load(sys.argv[1]))
U_model.load_state_dict(torch.load(sys.argv[2]))
f_pred = []
u_pred = []
for s in seq:
    s = s.unsqueeze(dim=0)
    f = F_model(s, z).squeeze().cpu().detach().numpy()
    u = U_model(s, z).squeeze().cpu().detach().numpy()
    f_pred.append(f)
    u_pred.append(u)
f_pred = list(map(list, zip(*f_pred)))
u_pred = list(map(list, zip(*u_pred)))

with open('F_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in f_pred:
        writer.writerow(row)
with open('U_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in u_pred:
        writer.writerow(row)

Structure_model = model(480)
Structure_model.load_state_dict(torch.load(sys.argv[3]))
structure_pred = []
for s in seq:
    s = s.unsqueeze(dim=0)
    structure = Structure_model(s, r).squeeze().cpu().detach().numpy()
    structure_pred.append(structure)
structure_pred = list(map(list, zip(*structure_pred)))
with open('Stru_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in structure_pred:
        writer.writerow(row)

# analyse
seq1 = torch.tensor([1] * 6 + [0] * 8 + [1] * 6).unsqueeze(dim=0)
seq2 = torch.tensor([0] * 6 + [1] * 8 + [0] * 6).unsqueeze(dim=0)
seq3 = torch.tensor([1] * 10 + [0] * 10).unsqueeze(dim=0)
seq4 = torch.tensor([1] * 20).unsqueeze(dim=0)
seq5 = torch.tensor([0] * 20).unsqueeze(dim=0)
strings = ['11111100000000111111', '00000011111111000000', '11111111110000000000',
           '11111111111111111111', '00000000000000000000']

test_seq = [seq1, seq2, seq3, seq4, seq5]
f = []
u = []
stru = []
for i in range(5):
    s = test_seq[i]
    f.append(F_model(s, z).squeeze().cpu().detach().numpy())
    u.append(U_model(s, z).squeeze().cpu().detach().numpy())
    stru.append(Structure_model(s, r).squeeze().cpu().detach().numpy())

for i in range(5):
    plt.plot(np.arange(-24, 24), f[i], label=strings[i])
plt.xlabel("z_dis")
plt.ylabel("F energy")
plt.title("F prediction")
plt.legend(loc='upper right')
plt.show()

for i in range(5):
    plt.plot(np.arange(-24, 24), u[i], label=strings[i])
plt.xlabel("z_dis")
plt.ylabel("U energy")
plt.title("U prediction")
plt.legend(loc='upper right')
plt.show()

for i in range(5):
    plt.plot(np.arange(0, 48, 0.1), stru[i], label=strings[i])
plt.xlabel("r_dis")
plt.ylabel("position")
plt.title("Struction prediction")
plt.legend(loc='upper right')
plt.show()