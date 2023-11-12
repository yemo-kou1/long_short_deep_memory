import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from datas.dataloader_factory import EarlyStopping
from odeLstm import TestODE, MultODE, ActivateLSDM, MultilayerLSDM, NODEFunc
from selfNODE import NeuralODE


def split_train_val_test():
    with open('E:/neuraldiffeq/eigenworms/data/processed/UEA/EigenWorms/data.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('E:/neuraldiffeq/eigenworms/data/processed/UEA/EigenWorms/labels.pkl', 'rb') as f:
        label = pickle.load(f)

    train_val_data, test_data, train_val_label, test_label = train_test_split(data, label, test_size=0.15,
                                                                              random_state=42)
    train_data, val_data, train_label, val_label = train_test_split(train_val_data, train_val_label, test_size=0.17,
                                                                    random_state=42)
    return train_data, train_label, val_data, val_label, test_data, test_label


def remove_first(dl):
    pre, target = dl.split([1, 6], 2)
    return target


def label_factory(labels):
    labels_pro = []
    for each in labels:
        # label = torch.Tensor([-1, -1, -1, -1, -1])
        label = torch.Tensor([0, 0, 0, 0, 0])
        label[each.int()] = 1
        labels_pro.append(label)
    # return torch.tensor(np.array(labels_pro), dtype=torch.float32)
    return torch.tensor([item.detach().numpy() for item in labels_pro])


def read_data():
    with open('datas/train_data.pkl', 'rb') as f:
        train_d = pickle.load(f)
    with open('datas/train_label.pkl', 'rb') as f:
        train_l = pickle.load(f)
    with open('datas/val_data.pkl', 'rb') as f:
        val_d = pickle.load(f)
    with open('datas/val_label.pkl', 'rb') as f:
        val_l = pickle.load(f)
    with open('datas/test_data.pkl', 'rb') as f:
        test_d = pickle.load(f)
    with open('datas/test_label.pkl', 'rb') as f:
        test_l = pickle.load(f)

    # print(train_d.size(), val_d.size(), test_d.size())
    # return remove_first(train_d), train_l, remove_first(val_d), val_l, remove_first(test_d), test_l
    return train_d, train_l, val_d, val_l, test_d, test_l


# 训练NODE的数据
def get_node_data(train_data):
    '''
    :param train_data:
    :return: list of data and label for NODE train
    '''
    Node_data_list = [0]
    Node_label_list = []
    for each in train_data:
        for da in each:
            Node_data_list.append(da)
            Node_label_list.append(da)

    del Node_data_list[0]
    del Node_label_list[0]

    return Node_data_list, Node_label_list


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
            Node_label_list.append(each[i + 1])
            if i + 2 == len(each):
                break

    print(len(Node_data_list))
    node_ds = ListDataSet(Node_data_list, Node_label_list)
    dl = dataloader.DataLoader(dataset=node_ds, batch_size=10000, shuffle=True)
    return dl


def squeeze_node(x, y):
    # x = torch.tensor(x)
    train_return = x[0].unsqueeze(0)
    # train_y_return = []
    num_counter = 15
    class_counter = [3, 3, 3, 3, 3]
    for i in range(len(y)):
        if class_counter[y[i]] != 0:
            train_return = torch.cat((train_return, x[i].unsqueeze(0)), dim=0)
            class_counter[y[i]] -= 1
            num_counter -= 1
            # train_y_return.append(train_y[i])
        else:
            continue

        if num_counter == 0:
            break
    train_return = train_return[1:]

    train_dl = get_node_dl(train_return)

    return train_dl


class ListDataSet(dataset.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len


def new_neural_ode_train(time_data, val_dl, model):
    loss_function = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=11,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # save_path = 'TestODE_7_32.pth'
    # early_stopping = EarlyStopping(patience=20, save_model=False, path=save_path)
    epochs = 50
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

            scheduler.step(avg_val_loss)

            # early_stopping(avg_val_loss, model)
            # if early_stopping.early_stop:
            #     print("微分方程训练完成！")
            #     print('train loss: ', avg_train_loss, 'val loss:', avg_val_loss)
            #     break

        if epoch % 5 == 0:
            print('train loss: ', avg_train_loss, 'val loss:', avg_val_loss,
                  'learn rate:', optimizer.state_dict()['param_groups'][0]['lr'])


def neural_ode_train(time_data, model):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 10

    for epoch in tqdm(range(epochs)):
        for data, label in time_data:
            optimizer.zero_grad()
            y_pred = model(data)

            label = label_factory(label)
            single_loss = loss_function(y_pred, label)
            single_loss.backward()
            optimizer.step()

        if epoch % 5 == 1:
            print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')


def solo_model_train(model, train_dl, val_dl, lr=0.0001):
    loss_f = nn.MSELoss()
    # loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # param_list = [{'params': each.parameters(), 'lr': lr} for each in model.hidden_lsdm]
    # param_list.append({'params': model.input.parameters(), 'lr': lr})
    # # param_list.append({'params': model.output.parameters(), 'lr': lr})
    # param_list.append({'params': model.output_layer.parameters(), 'lr': lr*10})
    # optimizer = torch.optim.Adam(param_list)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=20, min_lr=0, eps=1e-05)

    early_stopping = EarlyStopping(patience=70)  # , save_model=True, path=save_path)

    epochs = 1000
    best_score = 0
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


        model.eval()
        with torch.no_grad():
            valid_loss_list = []

            for valid_x, val_y in val_dl:
                valid_y = label_factory(val_y)

                outputs = model(valid_x)
                loss = loss_f(outputs, valid_y)
                valid_loss_list.append(loss.item())
            avg_valid_loss = np.average(valid_loss_list)

            early_stopping(avg_valid_loss, model)
            if early_stopping.early_stop:
                print("reach early stopping condition, training over！")
                model.load_state_dict(early_stopping.best_model.state_dict())
                break

            # scheduler.step(avg_valid_loss)

        score = test_solo_model_with_ncde(model, val_dl)
        if score > best_score:
            best_score = score
        if epoch % 10 == 0:
            print('train_loss: ', avg_train_loss,
                  'learn rate:', optimizer.state_dict()['param_groups'][0]['lr'],
                  '  |  ', 'train acc: ', test_solo_model_with_ncde(model, train_dl), '%',
                  '  |  ', 'val loss: ', avg_valid_loss,
                  '  |  ', 'val acc: ', score, '%', '  |  best val acc: ', best_score, '%')


def train_with_SGD(model, train_dl, val_dl):
    loss_f = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=12,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-05)

    # save_path = 'hidden70_.pth'
    early_stopping = EarlyStopping(patience=50)  # , save_model=True, path=save_path)

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

        model.eval()
        with torch.no_grad():
            valid_loss_list = []

            for valid_x, val_y in val_dl:
                valid_y = label_factory(val_y)

                outputs = model(valid_x)
                loss = loss_f(outputs, valid_y)
                valid_loss_list.append(loss.item())
            avg_valid_loss = np.average(valid_loss_list)

            # val_scheduler.step(avg_valid_loss)

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


def lstm_ode_train(model, train_dl, val_dl):
    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    for epoch in tqdm(range(epochs)):
        for data, label in train_dl:
            optimizer.zero_grad()
            y_pred = model(data)
            single_loss = loss_f(y_pred, label)
            single_loss.backward()
            optimizer.step()

        if epoch % 10 == 1:
            print('epoch : ', epoch, '  |  ',
                  'Train loss : ', single_loss.item(), '  |  ',
                  'val acc: ', test_solo_model_with_ncde(model, val_dl), '%')


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


def save_model_train(model, train_dl, val_dl, lr=0.0006, save_log='best_ML32204_model.pth'):
    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # param_list = [{'params': each.parameters(), 'lr': lr} for each in model.hidden_lsdm]
    # param_list.append({'params': model.input.parameters(), 'lr': lr})
    # param_list.append({'params': model.output_layer.parameters(), 'lr': lr * 100})
    # optimizer = torch.optim.Adam(param_list)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    early_stopping = EarlyStopping(patience=70)

    epochs = 1000
    best_score = 0
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

        model.eval()
        with torch.no_grad():
            valid_loss_list = []

            for valid_x, val_y in val_dl:
                valid_y = label_factory(val_y)

                outputs = model(valid_x)
                loss = loss_f(outputs, valid_y)
                valid_loss_list.append(loss.item())
            avg_valid_loss = np.average(valid_loss_list)

            early_stopping(avg_valid_loss, model)
            if early_stopping.early_stop:
                print("reach early stopping condition, training over！")
                model.load_state_dict(early_stopping.best_model.state_dict())
                break

            # scheduler.step(avg_valid_loss)

        score = test_solo_model_with_ncde(model, val_dl)
        if score > best_score:
            torch.save(model.state_dict(), save_log)
            best_score = score

        if epoch % 10 == 0:
            print('train_loss: ', avg_train_loss,
                  'learn rate:', optimizer.state_dict()['param_groups'][0]['lr'],
                  '  |  ', 'train acc: ', test_solo_model_with_ncde(model, train_dl), '%',
                  '  |  ', 'val loss: ', avg_valid_loss,
                  '  |  ', 'val acc: ', score, '%', '  |  best val acc: ', best_score, '%')


def train_with_cross(model, train_dl, val_dl, lr=0.0001):
    # loss_f = nn.MSELoss()
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # param_list = [{'params': each.parameters(), 'lr': lr} for each in model.hidden_lsdm]
    # param_list.append({'params': model.input.parameters(), 'lr': lr})
    # # param_list.append({'params': model.output.parameters(), 'lr': lr})
    # param_list.append({'params': model.output_layer.parameters(), 'lr': lr*10})
    # optimizer = torch.optim.Adam(param_list)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=20, min_lr=0, eps=1e-05)

    early_stopping = EarlyStopping(patience=70)  # , save_model=True, path=save_path)

    epochs = 1000
    best_score = 0
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss_list = []

        for data, train_y in train_dl:
            # label = label_factory(train_y)

            optimizer.zero_grad()
            y_pred = model(data)
            single_loss = loss_f(y_pred, train_y)
            train_loss_list.append(single_loss.item())
            single_loss.backward()
            optimizer.step()
        avg_train_loss = np.average(train_loss_list)
        scheduler.step(avg_train_loss)


        model.eval()
        with torch.no_grad():
            valid_loss_list = []

            for valid_x, val_y in val_dl:
                # valid_y = label_factory(val_y)

                outputs = model(valid_x)
                loss = loss_f(outputs, val_y)
                valid_loss_list.append(loss.item())
            avg_valid_loss = np.average(valid_loss_list)

            early_stopping(avg_valid_loss, model)
            if early_stopping.early_stop:
                print("reach early stopping condition, training over！")
                model.load_state_dict(early_stopping.best_model.state_dict())
                break

            # scheduler.step(avg_valid_loss)

        score = test_solo_model_with_ncde(model, val_dl)
        if score > best_score:
            best_score = score
        if epoch % 10 == 0:
            print('train_loss: ', avg_train_loss,
                  'learn rate:', optimizer.state_dict()['param_groups'][0]['lr'],
                  '  |  ', 'train acc: ', test_solo_model_with_ncde(model, train_dl), '%',
                  '  |  ', 'val loss: ', avg_valid_loss,
                  '  |  ', 'val acc: ', score, '%', '  |  best val acc: ', best_score, '%')


def main():
    train_d, train_l, val_d, val_l, test_d, test_l = read_data()

    # create data set
    train_data_set = ListDataSet(train_d, train_l)
    val_data_set = ListDataSet(val_d, val_l)
    test_data_set = ListDataSet(test_d, test_l)

    train_dl = dataloader.DataLoader(dataset=train_data_set, batch_size=181, shuffle=True)
    val_dl = dataloader.DataLoader(dataset=val_data_set, batch_size=39, shuffle=True)
    test_dl = dataloader.DataLoader(dataset=test_data_set, batch_size=39, shuffle=True)
    # train_dl, test_dl = merge_train_val()

    # Node_train_dl = get_node_dl(train_d)
    # Node_train_dl = squeeze_node(train_d, train_l)
    # Node_val_dl = get_node_dl(val_d)

    # Node = NeuralODE(TestODE(7, 32))
    # Node.load_state_dict(torch.load('TestODE_7_32.pth'))
    # new_neural_ode_train(Node_train_dl, Node_val_dl, Node)
    # Node.eval()

    lstm_model = MultilayerLSDM(7, 5, [16, 32, 32], [1, 5])
    # lstm_model = LSDM_for_long_seq(7, 14, 32, 5, Node, train_node=False, using_lsdm=True,
    #                                hidden_ode=NeuralODE(MultODE(14, 20)))
    # lstm_model = DoubleLayerLSDM(7, param1, 5, Node, train_node=False, hidden_ode=NeuralODE(TestODE(param1, 20)))
    # lstm_ode_train(lstm_model, train_dl, val_dl)
    save_model_train(lstm_model, train_dl, val_dl, lr=0.006)
    lstm_model.load_state_dict(torch.load('best_ML32204_model.pth'))
    print('train hit rate:', test_solo_model_with_ncde(lstm_model, train_dl), '%')
    print('val hit rate:', test_solo_model_with_ncde(lstm_model, val_dl), '%')
    print('test acc:', test_solo_model_with_ncde(lstm_model, test_dl), '%')


def train_multilayerLSDM(param_list, using_linear=False):
    train_d, train_l, val_d, val_l, test_d, test_l = read_data()

    # create data set
    train_data_set = ListDataSet(train_d, train_l)
    val_data_set = ListDataSet(val_d, val_l)
    test_data_set = ListDataSet(test_d, test_l)

    train_dl = dataloader.DataLoader(dataset=train_data_set, batch_size=181, shuffle=True)
    val_dl = dataloader.DataLoader(dataset=val_data_set, batch_size=39, shuffle=True)
    test_dl = dataloader.DataLoader(dataset=test_data_set, batch_size=39, shuffle=True)

    lstm_model = MultilayerLSDM(7, 5, param_list, using_linear)
    solo_model_train(lstm_model, train_dl, val_dl)
    print('train hit rate:', test_solo_model_with_ncde(lstm_model, train_dl), '%')
    print('val hit rate:', test_solo_model_with_ncde(lstm_model, val_dl), '%')
    print('test acc:', test_solo_model_with_ncde(lstm_model, test_dl), '%')


def merge_train_val(sample=True):
    if sample:
        train_d, train_l, val_d, val_l, test_d, test_l = read_data()
    else:
        train_d, train_l, val_d, val_l, test_d, test_l = split_train_val_test()
    # print(val_d.size(), val_l.size())
    # print(test_d.size(), test_l.size())
    train_val_d = torch.cat((train_d, val_d), 0)
    train_val_l = torch.cat((train_l, val_l), 0)
    # print(train_val_d.size())
    # print(train_val_l.size())

    # create data set
    train_val_data_set = ListDataSet(train_val_d, train_val_l)
    test_data_set = ListDataSet(test_d, test_l)

    train_val_dl = dataloader.DataLoader(dataset=train_val_data_set, batch_size=220, shuffle=True)
    test_dl = dataloader.DataLoader(dataset=test_data_set, batch_size=39, shuffle=True)

    return train_val_dl, test_dl


def merge_training(num_list, node_list, lr):
    print(num_list, node_list, lr)
    train_val_dl, test_dl = merge_train_val()

    # lstm_model = MultilayerLSDM(7, 5, num_list, dropout=False, node_layer=node_list)
    lstm_model = ActivateLSDM(7, 80, 5, ode=NODEFunc(7, 35, 4))
    # save_model_train(lstm_model, train_val_dl, test_dl, lr=0.006)
    # lstm_model.load_state_dict(torch.load('best_model.pth'))
    # train_with_cross(lstm_model, train_val_dl, test_dl, lr)
    solo_model_train(lstm_model, train_val_dl, test_dl, lr)
    print('train hit rate:', test_solo_model_with_ncde(lstm_model, train_val_dl), '%')
    print('test acc:', test_solo_model_with_ncde(lstm_model, test_dl), '%')


def model_result(address):
    train_d, train_l, val_d, val_l, test_d, test_l = read_data()

    # create data set
    train_data_set = ListDataSet(train_d, train_l)
    val_data_set = ListDataSet(val_d, val_l)
    test_data_set = ListDataSet(test_d, test_l)

    train_dl = dataloader.DataLoader(dataset=train_data_set, batch_size=181, shuffle=True)
    val_dl = dataloader.DataLoader(dataset=val_data_set, batch_size=39, shuffle=True)
    test_dl = dataloader.DataLoader(dataset=test_data_set, batch_size=39, shuffle=True)

    node = NeuralODE(TestODE(7, 32))
    node.eval()
    model = ActivateLSDM(7, 32, 5, node, train_node=False)
    model.load_state_dict(torch.load(address))
    print('train hit rate:', test_solo_model_with_ncde(model, train_dl), '%')
    print('val hit rate:', test_solo_model_with_ncde(model, val_dl), '%')
    print('test hit rate:', test_solo_model_with_ncde(model, test_dl, True), '%')


if __name__ == '__main__':
    main()
