import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import dataloader
from tqdm import tqdm

import controldiffeq
from experiments.datasets.sepsis import get_data
from experiments.odelstm.datas.dataloader_factory import EarlyStopping, ListDataSet, TensorDataSet
from experiments.odelstm.odeLstm import TestODE, LSTMode, MultODE, LSTM_with_NODE, ActivateLSDM, NODEFunc, LSDM, ODE, \
    MultilayerLSDM, LSDMModule, GRU, GRU_ODE
from experiments.odelstm.selfNODE import NeuralODE


def label_factory(labels):
    labels_pro = []
    for each in labels:
        label = [-1, -1]
        label[each.int()] = 1
        labels_pro.append(label)
    return torch.tensor(labels_pro, dtype=torch.float32)


def get_train_dataloader():
    train_data = torch.load(loc + 'train_a.pt')
    val_data = torch.load(loc + 'val_a.pt')
    test_data = torch.load(loc + 'test_a.pt')
    # print(train_data.size())

    # train_transform = normalization(train_data)
    # val_tran = normalization(val_data)
    # test_tran = normalization(test_data)

    train_y = torch.load(loc + 'train_y.pt')
    val_y = torch.load(loc + 'val_y.pt')
    test_y = torch.load(loc + 'test_y.pt')

    train_transform = torch.cat((train_data, val_data), 0)
    train_y = torch.cat((train_y, val_y), 0)
    print(train_transform.size(), val_data.size(), test_data.size())

    train_ds = TensorDataSet(train_transform, train_y)
    val_ds = TensorDataSet(val_data, val_y)
    test_ds = TensorDataSet(test_data, test_y)

    train_dl = dataloader.DataLoader(dataset=train_ds, batch_size=1024, shuffle=True)
    val_dl = dataloader.DataLoader(dataset=val_ds, batch_size=1024, shuffle=True)
    test_dl = dataloader.DataLoader(dataset=test_ds, batch_size=1024, shuffle=True)

    return train_dl, val_dl, test_dl


def get_train_dataloader_with_oi():
    train_data = torch.load(loc + 'train_a.pt')
    val_data = torch.load(loc + 'val_a.pt')
    test_data = torch.load(loc + 'test_a.pt')

    train_static = torch.load(loc + 'train_static.pt')
    val_static = torch.load(loc + 'val_static.pt')
    test_static = torch.load(loc + 'test_static.pt')

    train_transform = normalization(train_data)
    val_tran = normalization(val_data)
    test_tran = normalization(test_data)

    train_y = torch.load(loc + 'train_y.pt')
    val_y = torch.load(loc + 'val_y.pt')
    test_y = torch.load(loc + 'test_y.pt')

    train_ds = torch.utils.data.TensorDataset(train_transform, train_static, train_y)
    val_ds = torch.utils.data.TensorDataset(val_tran, val_static, val_y)
    test_ds = torch.utils.data.TensorDataset(test_tran, test_static, test_y)

    train_dl = dataloader.DataLoader(dataset=train_ds, batch_size=1024, shuffle=True)
    val_dl = dataloader.DataLoader(dataset=val_ds, batch_size=1024, shuffle=True)
    test_dl = dataloader.DataLoader(dataset=test_ds, batch_size=1024, shuffle=True)

    return train_dl, val_dl, test_dl


# 训练NODE的数据
def get_node_dl(train_data):
    """
    :param train_data:
    :return: list of data and label for NODE train
    """
    Node_data_list = []
    Node_label_list = []
    for each in train_data:
        for i in range(len(each)):
            Node_data_list.append(each[i])
            Node_label_list.append(each[i+1])
            if i+2 == len(each):
                break

    # print(len(Node_data_list))
    node_ds = ListDataSet(Node_data_list, Node_label_list)
    dl = dataloader.DataLoader(dataset=node_ds, batch_size=1024, shuffle=True)
    return dl


loc = 'E:/neuraldiffeq/speech_and_physionet' \
      '/experiments/datasets/processed_data/sepsis_nostaticintensity_notimeintensity/'


def node_train_val():
    train_d = torch.load(loc + 'test_a.pt')
    val_d = torch.load(loc + 'val_a.pt')

    train_dl = get_node_dl(train_d)
    val_dl = get_node_dl(val_d)

    return train_dl, val_dl


def squeeze_node():
    train_y = torch.load(loc + 'train_y.pt')
    val_y = torch.load(loc + 'val_y.pt')

    train_data = torch.load(loc + 'train_a.pt')
    # train_data = normalization(train_data)
    train_return = train_data[0].unsqueeze(0)
    # train_y_return = []
    num_counter = 200
    for i in range(len(train_y)):
        train_return = torch.cat((train_return, train_data[i].unsqueeze(0)), dim=0)
        # train_y_return.append(train_y[i])
        num_counter -= 1
        if num_counter == 0:
            break
    train_return = normalization(train_return[1:])
    # print(train_return.size())
    # train_y_return = train_y_return[1:]
    # print(train_y_return, len(train_y_return))

    val_data = torch.load(loc + 'val_a.pt')
    # val_data = normalization(val_data)

    val_return = val_data[0].unsqueeze(0)
    num_counter = 30
    for i in range(len(val_y)):
        val_return = torch.cat((val_return, val_data[i].unsqueeze(0)), dim=0)
        num_counter -= 1
        if num_counter == 0:
            break
    val_return = normalization(val_return[1:])

    train_dl = get_node_dl(train_return)
    val_dl = get_node_dl(val_return)

    return train_dl, val_dl


def neural_ode_train(time_data, val_dl, model):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=5, min_lr=0, eps=1e-08)
    early_stopping = EarlyStopping(patience=20)
    epochs = 200
    for epoch in tqdm(range(epochs)):
        train_loss = []
        model.train()
        for data, label in time_data:
            optimizer.zero_grad()
            y_pred = model(data)

            single_loss = loss_function(y_pred, label)
            train_loss.append(single_loss.item())
            single_loss.backward()
            optimizer.step()
        avg_train_loss = np.average(train_loss)
        scheduler.step(avg_train_loss)

        model.eval()
        with torch.no_grad():
            valid_loss = []
            for val_d, val_l in val_dl:
                output = model(val_d)
                loss = loss_function(output, val_l)
                valid_loss.append(loss.item())
            avg_val_loss = np.average(valid_loss)

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("微分方程训练完成！")
                print('train loss: ', avg_train_loss, 'val loss:', avg_val_loss)
                break

        if epoch % 5 == 0:
            print('train loss: ', avg_train_loss,  'val loss:', avg_val_loss,
                  'learn rate:', optimizer.state_dict()['param_groups'][0]['lr'])


def solo_model_train(model, train_dl, val_dl, save_log, lr=0.0032):
    loss_f = nn.MSELoss()
    # lr = 0.0032
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam([{'params': model.lstm.parameters(), 'lr': lr},
    #                               {'params': model.outer.parameters(), 'lr': lr*100}])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # save_path = 'hidden140_.pth'
    early_stopping = EarlyStopping(patience=70)  # , save_model=True, path=save_path)
    rate = 0
    epochs = 300
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss_list = []
        for data, train_y in train_dl:
            # label = label_factory(train_y)
            optimizer.zero_grad()

            y_pred = model(data).squeeze(-1)
            single_loss = loss_f(y_pred, train_y)
            train_loss_list.append(single_loss.item())
            single_loss.backward()
            optimizer.step()
        avg_train_loss = np.average(train_loss_list)
        scheduler.step(avg_train_loss)

        model.eval()
        with torch.no_grad():
            valid_loss_list = []

            for valid_x, valid_y in val_dl:
                outputs = model(valid_x).squeeze(-1)
                loss = loss_f(outputs, valid_y)
                valid_loss_list.append(loss.item())
            avg_valid_loss = np.average(valid_loss_list)

            early_stopping(avg_valid_loss, model)
            if early_stopping.early_stop:
                print("reach early stopping condition, training over！")
                break

        val_rate = com_auc_with_score(model, val_dl)

        if val_rate > rate:
            # torch.save(model.state_dict(), save_log)
            rate = val_rate
            # counter = 0

        # if val_rate <= rate:
        #     counter += 1
        #     if counter > 30:
        #         print("reach early stopping condition, training over！")
        #         break

        if epoch % 5 == 0:
            print('train_loss: ', avg_train_loss,
                  'learn rate:', optimizer.state_dict()['param_groups'][0]['lr'],
                  '  |  ', 'train auc: ', com_auc_with_score(model, train_dl)*100, '%',
                  '  |  ', 'val loss: ', avg_valid_loss,
                  '  |  ', 'val auc: ', val_rate*100, '%', '  |  best val auc: ', rate*100, '%')


def test_solo_model_with_ncde(model, dl, info=False):
    model.eval()
    num_hitting = 0.0
    size = 0.0
    with torch.no_grad():

        for data, train_y in dl:
            predict = model(data).squeeze(-1)
            # pre_label = torch.max(predict, 1).indices
            for i in range(len(train_y)):
                if predict[i] > 0.5 and train_y[i] == 1:
                    num_hitting += 1
                if predict[i] < 0.5 and train_y[i] == 0:
                    num_hitting += 1
                if info:
                    print(predict[i], train_y[i], num_hitting)
                size += 1

    return num_hitting / size * 100


def com_auc(model, dl):
    model.eval()
    with torch.no_grad():
        for data, label in dl:
            pre_x = model(data)
            predict = torch.max(pre_x, 1).indices

            if 'predict_label' in locals().keys():
                predict_label = torch.cat((predict_label, predict))
                true_label = torch.cat((true_label, label))

            else:
                predict_label = predict
                true_label = label

    # print(predict_label.size())
    auc = roc_auc_score(true_label, predict_label)
    return auc


def com_auc_with_score(model, dl):
    model.eval()
    with torch.no_grad():
        for data, label in dl:
            pre_x = model(data).squeeze(-1)

            if 'predict_score' in locals().keys():
                predict_score = torch.cat((predict_score, pre_x))
                true_label = torch.cat((true_label, label))

            else:
                predict_score = pre_x
                true_label = label
    auc = roc_auc_score(true_label, predict_score)
    return auc


def train_solo_floor():  # model):
    # save_log = 'sepsis_' + datetime.datetime.now().strftime('%y_%m_%d_%H_%M') + '.pth'
    save_log = 'sepsis_model.pth'
    train_dl, val_dl, test_dl = get_train_dataloader()
    # times, train_dl, val_dl, test_dl = get_data(False, False, batch_size=1024)

    # node = ODE(35, 10)
    # node = NeuralODE(TestODE(35, 70))
    # node = NeuralODE(MultODE(35, 120))
    # # node = NeuralODE(GRU(35))
    #
    # node_train_dl, node_val_dl = squeeze_node()
    # # node_train_dl, node_val_dl = node_train_val()
    # neural_ode_train(node_train_dl, node_val_dl, node)
    # node.eval()

    # model = ActivateLSDM(35, 120, 1, node, train_node=False, dropout=False)
    model = ActivateLSDM(35, 120, 1, NeuralODE(MultODE(35, 120)), train_node=True, dropout=False)
    # model = MultilayerLSDM(35, 1, [70])
    # model = MultilayerLSDM(35, 1, [64, 16, 4])
    solo_model_train(model, train_dl, test_dl, save_log, 0.0064)
    # model.load_st`ate_dict(torch.load(save_log))

    print('train auc:', com_auc_with_score(model, train_dl) * 100)
    print('val auc:', com_auc_with_score(model, val_dl) * 100)
    print('test auc:', com_auc_with_score(model, test_dl) * 100)


def model_result(address):
    train_dl, val_dl, test_dl = get_train_dataloader()

    node = NeuralODE(GRU(35))
    node.eval()
    model = ActivateLSDM(35, 70, 1, node, train_node=True, dropout=True)
    model.load_state_dict(torch.load(address))
    # print('train auc:', com_auc(model, train_dl)*100)
    # print('val auc:', com_auc(model, val_dl)*100)
    # print('test auc:', com_auc(model, test_dl)*100)
    print('--------------------------------------------')
    print('train auc:', com_auc_with_score(model, train_dl) * 100)
    print('val auc:', com_auc_with_score(model, val_dl) * 100)
    print('test auc:', com_auc_with_score(model, test_dl) * 100)


if __name__ == '__main__':
    for i in range(5):
        train_solo_floor()


