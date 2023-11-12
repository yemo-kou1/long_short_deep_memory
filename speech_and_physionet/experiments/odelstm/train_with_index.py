import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import dataloader
from tqdm import tqdm

from experiments.odelstm.datas.dataloader_factory import EarlyStopping, ListDataSet, label_factory, TensorDataSet
from experiments.odelstm.odeLstm import TestODE, LSTMode, MultODE, ActivateLSDM
from experiments.odelstm.selfNODE import NeuralODE


def normalization(data):
    sc = MinMaxScaler(feature_range=(-1, 1))
    transform = []
    for each in data:
        sc.fit(each)
        tran = sc.transform(each).astype(np.float32)
        transform.append(tran)

    return transform


def get_train_dataloader():
    train_data = torch.load(loc + 'train_a.pt')
    val_data = torch.load(loc + 'val_a.pt')
    test_data = torch.load(loc + 'test_a.pt')

    print(train_data.size(), val_data.size())

    train_transform = normalization(train_data)
    val_tran = normalization(val_data)
    test_tran = normalization(test_data)

    train_y = torch.load(loc + 'train_y.pt')
    val_y = torch.load(loc + 'val_y.pt')
    test_y = torch.load(loc + 'test_y.pt')

    train_ds = TensorDataSet(train_transform, train_y)
    val_ds = TensorDataSet(val_tran, val_y)
    test_ds = TensorDataSet(test_tran, test_y)

    train_dl = dataloader.DataLoader(dataset=train_ds, batch_size=1024, shuffle=True)
    val_dl = dataloader.DataLoader(dataset=val_ds, batch_size=1024, shuffle=True)
    test_dl = dataloader.DataLoader(dataset=test_ds, batch_size=1024, shuffle=True)

    return train_dl, val_dl, test_dl


# 训练NODE的数据
def get_node_dl(train_data):
    '''
    :param train_data:
    :return: list of data and label for NODE train
    '''
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
      '/experiments/datasets/processed_data/speech_commands_with_mels/'


def get_solo_node_data():
    train_d = torch.load(loc + 'train_a.pt')
    val_d = torch.load(loc + 'val_a.pt')

    train_dl = get_node_dl(train_d)
    val_dl = get_node_dl(val_d)

    return train_dl, val_dl


def squeeze_node(seq):
    counter = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    data_str = ['a', 'b', 'c', 'd']
    train_y = torch.load(loc + 'train_y.pt')
    val_y = torch.load(loc + 'val_y.pt')

    train_data = torch.load(loc + 'train_' + data_str[seq] + '.pt')
    train_return = train_data[0].unsqueeze(0)
    # train_y_return = []
    num_counter = 100
    for i in range(len(train_y)):
        if counter[train_y[i]] != 0:
            train_return = torch.cat((train_return, train_data[i].unsqueeze(0)), dim=0)
            # train_y_return.append(train_y[i])
            num_counter -= 1
        if num_counter == 0:
            break
    train_return = normalization(train_return[1:])

    val_data = torch.load(loc + 'val_' + data_str[seq] + '.pt')

    val_return = val_data[0].unsqueeze(0)
    val_counter = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    num_counter = 30
    for i in range(len(val_y)):
        if val_counter[val_y[i]] != 0:
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    early_stopping = EarlyStopping(patience=8)
    epochs = 100
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

            scheduler.step(avg_val_loss)

        if epoch % 5 == 0:
            print('train loss: ', avg_train_loss, 'val loss:', avg_val_loss,
                  'learn rate:', optimizer.state_dict()['param_groups'][0]['lr'])


def solo_model_train(model, train_dl, val_dl, save_log):
    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    early_stopping = EarlyStopping(patience=30)

    rate = 0
    counter = 0
    epochs = 200
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss_list = []
        for data, train_y in train_dl:
            label = label_factory(train_y)

            optimizer.zero_grad()
            y_pred = model(data)
            single_loss = loss_f(y_pred, label)
            train_loss_list.append(single_loss.item())
            single_loss.backward()
            optimizer.step()
        avg_train_loss = np.average(train_loss_list)
        scheduler.step(avg_train_loss)

        val_rate = test_solo_model_with_ncde(model, val_dl)

        if val_rate > rate:
            torch.save(model.state_dict(), save_log)
            rate = val_rate
            counter = 0

        # if val_rate <= rate:
        #     counter += 1
        #     if counter > 20:
        #         print("reach early stopping condition, training over！")
        #         break

        model.eval()
        with torch.no_grad():
            valid_loss_list = []
            for valid_x, train_y in val_dl:
                valid_y = label_factory(train_y)

                outputs = model(valid_x)
                loss = loss_f(outputs, valid_y)
                valid_loss_list.append(loss.item())
            avg_valid_loss = np.average(valid_loss_list)

            early_stopping(avg_valid_loss, model)
            if early_stopping.early_stop:
                print("reach early stopping condition, training over！")
                break

        if epoch % 10 == 0:
            print('train_loss: ', avg_train_loss,
                  'learn rate:', optimizer.state_dict()['param_groups'][0]['lr'],
                  '  |  ', 'train acc: ', test_solo_model_with_ncde(model, train_dl), '%',
                  '  |  ', 'val loss: ', avg_valid_loss,
                  '  |  ', 'val acc: ', test_solo_model_with_ncde(model, val_dl), '%')


def test_solo_model_with_ncde(model, dl, info=False):
    model.eval()
    num_hitting = 0.0
    size = 0.0
    with torch.no_grad():
        for data, train_y in dl:
            predict = model(data)
            pre_label = torch.max(predict, 1).indices
            for i in range(len(train_y)):
                if train_y[i] == pre_label[i]:
                    num_hitting += 1
                if info:
                    print(predict[i], pre_label[i], train_y[i], num_hitting)
                size += 1

    return num_hitting / size * 100


def train_solo_floor():
    save_log = datetime.datetime.now().strftime('%y_%m_%d_%H_%M') + '_116hidden_model.pth'
    train_dl, val_dl, test_dl = get_train_dataloader()

    node = NeuralODE(MultODE(21, 40))
    node_train_dl, node_val_dl = squeeze_node(0)
    neural_ode_train(node_train_dl, node_val_dl, node)
    node.eval()

    model = ActivateLSDM(21, 116, 10, node, train_node=False)
    # model = ActivateLSDM(21, 40, 10, node, train_node=False)
    solo_model_train(model, train_dl, val_dl, save_log)

    model.load_state_dict(torch.load(save_log))
    print('train hit rate:', test_solo_model_with_ncde(model, train_dl), '%')
    print('val hit rate:', test_solo_model_with_ncde(model, val_dl), '%')
    print('test hit rate:', test_solo_model_with_ncde(model, test_dl), '%')


def model_result(add):
    train_dl, val_dl, test_dl = get_train_dataloader()
    node = NeuralODE(MultODE(21, 40))
    model = LSTMode(21, 40, 10, node, train_node=False)
    model.load_state_dict(torch.load(add))
    print('train hit rate:', test_solo_model_with_ncde(model, train_dl), '%')
    print('val hit rate:', test_solo_model_with_ncde(model, val_dl), '%')
    print('test hit rate:', test_solo_model_with_ncde(model, test_dl), '%')


if __name__ == '__main__':
    for i in range(5):
        train_solo_floor()


