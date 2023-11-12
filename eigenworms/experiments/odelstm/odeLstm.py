import torch
from torch import nn

from torch.nn import Parameter

from experiments.odelstm.selfNODE import ODEF, NeuralODE
from ncdes.model import BaseModel


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
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.additional_layers = []
        for i in range(num_layers):
            self.additional_layers.append(nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)))

        for i in range(len(self.additional_layers)):
            self.add_module('layer_' + str(i), self.additional_layers[i])

        self.inner = nn.Linear(input_dim, hidden_dim)
        self.outer = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, input_dim))

    def forward(self, h, t):
        temp = self.inner(h)
        for each in self.additional_layers:
            temp = each(temp)
        return self.outer(temp)


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
        # self.Wnn = Parameter(torch.Tensor(hidden_sz, hidden_sz))
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


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index][:, -1]


class ActivateLSDM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, ode=None, train_node=True, dropout=False):
        super(ActivateLSDM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        if ode is None:
            ode = NeuralODE(TestODE(input_size, hidden_layer_size))
        self.lstm = ActivateLSDMModule(input_size, hidden_layer_size, ode, train_node)

        if dropout:
            self.outer = nn.Sequential(
                nn.Linear(hidden_layer_size, output_size),
                nn.Dropout(0.5)
            )
        else:
            self.outer = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = None

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.outer(lstm_out[:, -1])
        self.hidden_cell = None
        return predictions


# without output layer
class DoubleLayerLSDM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, ode=None,
                 train_node=True, hidden_ode=None):
        super(DoubleLayerLSDM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        if ode is None:
            ode = NeuralODE(TestODE(input_size, hidden_layer_size))
        # self.lstm = LSTM_with_NODE(input_size, hidden_layer_size, ode, train_node)
        self.lstm = ActivateLSDMModule(input_size, hidden_layer_size, ode, train_node)

        if hidden_ode is None:
            hidden_ode = NeuralODE(TestODE(hidden_layer_size, output_size))
        self.hidden_lstm = ActivateLSDMModule(hidden_layer_size, output_size, hidden_ode)

    @staticmethod
    def sample(lstm_result):
        num = lstm_result.size(1)
        sample_lv = int(num ** 0.5)
        sample_num = int(num / sample_lv)
        sample_ind = []
        for i in range(sample_num):
            sample_ind.append(i * sample_lv)
        sample_ind = torch.tensor(sample_ind, dtype=torch.long)

        sample_res = torch.index_select(lstm_result, 1, sample_ind)
        return sample_res

    def forward(self, input_seq):
        # 181, 4496, 6  =>  181, 4496, hidden size
        lstm_out, hidden0 = self.lstm(input_seq, None)
        sample = self.sample(lstm_out)

        predictions, hidden1 = self.hidden_lstm(sample, None)

        predictions = torch.softmax(predictions[:, -1], 1)
        return predictions


class TupleLSDM(nn.Module):
    def __init__(self, input_size, first_size, second_size, third_size, output_size, ode1=None, ode2=None,
                 train_node=True, hidden_ode=None):
        super(TupleLSDM, self).__init__()
        if ode1 is None:
            # Node的input_size要与lstm相同,中间的隐藏层可不同
            ode1 = NeuralODE(TestODE(input_size, first_size))
        # self.input = ActivateLSDMModule(input_size, first_size, ode1, train_node)
        self.input = LSTM_with_NODE(input_size, first_size, ode1, train_node)

        if ode2 is None:
            ode2 = NeuralODE(TestODE(first_size, second_size))
        # self.second = ActivateLSDMModule(first_size, second_size, ode2, True)
        self.second = LSTM_with_NODE(first_size, second_size, ode2, True)

        # self.output = ActivateLSDM(second_size, third_size, output_size, hidden_ode)
        self.output = LSTMode(second_size, third_size, output_size, hidden_ode)

        self.hidden_cell = None

    @staticmethod
    def sample(lstm_result):
        num = lstm_result.size(1)
        sample_lv = int(num ** (1 / 3))  # 采样率
        sample_num = int(num / sample_lv)
        sample_ind = []
        for i in range(sample_num):
            sample_ind.append(i * sample_lv)
        sample_ind = torch.tensor(sample_ind, dtype=torch.long)

        sample_res = torch.index_select(lstm_result, 1, sample_ind)
        return sample_res

    def forward(self, input_seq):
        # 181, 4496, 6  =>  181, 4496, hidden size
        lstm_out, self.hidden_cell = self.input(input_seq, self.hidden_cell)
        sample = self.sample(lstm_out)

        second_out, hidden_cell = self.second(sample, None)
        hidden = self.sample(second_out)

        predictions = self.output(hidden)

        self.hidden_cell = None
        return predictions


class MultilayerLSDM(BaseModel):
    def __init__(self, input_size, output_size, layer_param=None, dropout=False,
                 input_ode=None, train_ode=True, node_layer=None):
        super(MultilayerLSDM, self).__init__()
        if node_layer is None:
            node_layer = [1, 5]
        if layer_param is None:
            layer_param = [14, 32]
        self.layer = len(layer_param)
        # self.ODE_hidden_size = ode_layer * 4

        if input_ode is None:
            input_ode = NeuralODE(NODEFunc(input_size, layer_param[0], node_layer[0]))
        self.input = ActivateLSDMModule(input_size, layer_param[0], input_ode, train_node=train_ode)

        self.hidden_lsdm = []
        for i in range(self.layer - 1):
            hidden_ode = NeuralODE(NODEFunc(layer_param[i], layer_param[i + 1], node_layer[1]))
            hidden = ActivateLSDMModule(layer_param[i], layer_param[i+1], node=hidden_ode)

            self.add_module(f"hidden_lsdm_{i + 1}", hidden)
            self.hidden_lsdm.append(hidden)
        if dropout:
            self.output_layer = nn.Sequential(
                nn.Linear(layer_param[-1], output_size),
                nn.Dropout(0.3)
            )
        else:
            self.output_layer = nn.Linear(layer_param[-1], output_size)

    @staticmethod
    def sample(lstm_result, layer):
        num = lstm_result.size(1)
        sample_lv = int(num ** (1 / layer))
        sample_num = int(num / sample_lv)
        sample_ind = []
        for i in range(sample_num):
            sample_ind.append(i * sample_lv)
        sample_ind = torch.tensor(sample_ind, dtype=torch.long)

        sample_res = torch.index_select(lstm_result, 1, sample_ind)
        return sample_res

    def forward(self, input_seq):
        temp_out, hidden_cell = self.input(input_seq, None)

        for each in self.hidden_lsdm:
            temp_out = self.sample(temp_out, 2)
            temp_out, hidden_cell = each(temp_out, None)
        predictions = self.output_layer(temp_out[:, -1])

        return predictions


