import copy

import torch
from torch import nn
import math

from torch.nn import Parameter
from tqdm import tqdm

from torchdiffeq import odeint, odeint_adjoint

from experiments.datasets.speech_commands import get_data
from experiments.models import BaseModel
from experiments.odelstm.datas.dataloader_factory import label_factory, data_factory
from experiments.odelstm.selfNODE import ODEF, NeuralODE


class simpleODE(ODEF):
    def __init__(self):
        super(simpleODE, self).__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, y, x):
        return x


class TestODE(ODEF):
    def __init__(self, input_size, hidden_size):
        super(TestODE, self).__init__()
        # self.lin = nn.Linear(input_size, input_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, y, x):
        y = self.net(y)
        return y


class GRU(ODEF):
    def __init__(self, input_size):
        super(GRU, self).__init__()
        self.z_connect = nn.Linear(input_size, input_size)
        self.r_connect = nn.Linear(input_size, input_size)
        self.g_connect = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_size, input_size)
        )

    # y: 1*input_size
    def forward(self, y, x):
        z = torch.sigmoid(self.z_connect(y))
        r = torch.sigmoid(self.r_connect(y))
        g = self.g_connect(r * y)
        return (1 - z) * (g - y)


class GRU_ODE(ODEF):
    def __init__(self, input_size, hidden_size):
        super(GRU_ODE, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.z_connect = nn.Linear(input_size + hidden_size, hidden_size)
        self.r_connect = nn.Linear(input_size + hidden_size, hidden_size)
        self.g_connect = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.outer = nn.Linear(hidden_size, input_size)

    def forward(self, x, t):
        h = self.input(x)
        xh = torch.cat((x, h), dim=1)
        z = torch.sigmoid(self.z_connect(xh))
        r = torch.sigmoid(self.r_connect(xh))
        g = self.g_connect(r * h)
        return self.outer((1 - z) * (g - h))


class ODE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ODE, self).__init__()
        # self.lin = nn.Linear(input_size, input_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x, y):
        y = self.net(y)
        return y


class MultODE(ODEF):
    def __init__(self, input_size, hidden_size):
        super(MultODE, self).__init__()
        # self.lin = nn.Linear(input_size, input_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, y, x):
        y = self.net(y)
        return y


class NODEFunc(ODEF):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(NODEFunc, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Additional layers are just hidden to hidden with relu activation
        additional_layers = [nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 1) if num_layers > 1 else []

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            *additional_layers,
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, h, t):
        return self.net(h)


class ActivateLSDMModule(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, node: NeuralODE, train_node=True):
        super(ActivateLSDMModule, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.Node = node
        self.train_node = train_node
        # self.connect = nn.Linear(input_sz, hidden_sz)
        # Define/initialize all tensors
        # forget gate
        self.Wf = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bf = Parameter(torch.Tensor(hidden_sz))
        # input gate
        self.Wi = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bi = Parameter(torch.Tensor(hidden_sz))
        # Candidate memory cell
        self.Wc = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bc = Parameter(torch.Tensor(hidden_sz))
        # output gate
        self.Wo = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bo = Parameter(torch.Tensor(hidden_sz))

        # gate for nn and NODE connect
        self.Wnn = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.wode = Parameter(torch.Tensor(input_sz, hidden_sz))
        # self.bn = Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        # ht and Ct start as the previous states and end as the output states in each loop bellow
        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            hx_concat = torch.cat((ht, xt), dim=1)

            ### The LSTM Cell!
            ft = torch.sigmoid(hx_concat @ self.Wf + self.bf)
            it = torch.sigmoid(hx_concat @ self.Wi + self.bi)
            Ct_candidate = torch.tanh(hx_concat @ self.Wc + self.bc)

            if self.train_node:
                p = self.Node(xt)
            else:
                self.Node.eval()
                with torch.no_grad():
                    p = self.Node(xt)

            ot = torch.sigmoid(hx_concat @ self.Wo + p @ self.wode + self.bo)
            Ct = ft * Ct + it * Ct_candidate
            ht = ot * torch.tanh(Ct)

            hidden_seq.append(ht.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (ht, Ct)


class ActivateLSDM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, ode=None, train_node=True, dropout=False):
        super(ActivateLSDM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        if ode is None:
            # Node的input_size要与lstm相同,中间的隐藏层可不同
            ode = NeuralODE(TestODE(input_size, hidden_layer_size))
        self.lstm = ActivateLSDMModule(input_size, hidden_layer_size, ode, train_node)

        if dropout:
            self.outer = nn.Sequential(
                nn.Linear(hidden_layer_size, output_size),
                nn.Dropout(0.3)
            )
        else:
            self.outer = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = None

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)

        predictions = self.outer(lstm_out[:, -1])

        # predictions = nn.softmax(predictions, dim=-1)
        self.hidden_cell = None
        return predictions

