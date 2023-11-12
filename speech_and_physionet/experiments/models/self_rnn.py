import torch
from torch import nn

from torch.nn import Parameter
from torchdiffeq import odeint

from experiments.odelstm.selfNODE import NeuralODE


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

            # p = self.connect(p)

            ot = torch.sigmoid(hx_concat @ self.Wo + p @ self.wode + self.bo)
            # ot = ot @ self.Wnn  self.bn
            # outputs
            Ct = ft * Ct + it * Ct_candidate
            ht = ot * torch.tanh(Ct)

            hidden_seq.append(ht.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (ht, Ct)


class ActivateLSDM(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, ode=None, train_node=True, dropout=False):
        super(ActivateLSDM, self).__init__()
        self.hidden_layer_size = hidden_channels

        if ode is None:
            ode = NeuralODE(ODE(input_channels, hidden_channels))
        self.lstm = ActivateLSDMModule(input_channels, hidden_channels, ode, train_node)

        if dropout:
            self.outer = nn.Sequential(
                nn.Linear(hidden_channels, output_channels),
                nn.Dropout(0.3)
            )
        else:
            self.outer = nn.Linear(hidden_channels, output_channels)
        self.hidden_cell = None

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.outer(lstm_out[:, -1])
        self.hidden_cell = None
        return predictions

