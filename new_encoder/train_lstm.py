# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/11/3
Description:
"""
# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/11/3
Description:
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import logger_conf
from lstm_model import LSTM
from new_encoder.data_openml import data_prep_china_options, DataSetCatCon_2
from new_encoder.utils import classification_scores_2
from utils import get_scheduler


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    parser.add_argument('--gpu_index', default=7, type=int)
    parser.add_argument('--task', default='binary', type=str, choices=['binary', 'multiclass', 'regression'])
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--dset_seed', default=1, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])
    opt = parser.parse_args()
    return opt


def train(device):
    input_feature_size = 28
    hidden_size = 512
    num_layers = 10
    dropout = 0.
    num_classes = 2
    model = LSTM(input_feature_size,
                 hidden_size,
                 num_layers,
                 dropout, num_classes).to(device)
    # print(model)
    training_df, validation_df, testing_df, latest_df = data_prep_china_options(opt.dset_seed)
    train_ds = DataSetCatCon_2(training_df)
    # trainloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=int(cpu_count() * 0.7))
    trainloader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=int(cpu_count() * 0.7))

    valid_ds = DataSetCatCon_2(validation_df)
    validloader = DataLoader(valid_ds, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    test_ds = DataSetCatCon_2(testing_df)
    testloader = DataLoader(test_ds, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    latest_ds = DataSetCatCon_2(latest_df)
    latestloader = DataLoader(latest_ds, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = get_scheduler(opt, optimizer)
    best_valid_accuracy = 0
    best_test_accuracy = 0
    best_valid_accuracy_1 = 0
    best_valid_accuracy_2 = 0
    best_test_accuracy_1 = 0
    best_test_accuracy_2 = 0
    running_losss = []
    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0
        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            optimizer.zero_grad()
            x, y = data[0].to(device), data[1].long().to(device)
            predict_out = model(x)

            loss = criterion(predict_out, y.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        running_losss.append(np.array(running_loss).mean())
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():

                valid_auc, valid_acc, v_acc_1, v_acc_2 = classification_scores_2(model, validloader, device)
                test_auc, test_acc, t_acc_1, t_acc_2 = classification_scores_2(model, testloader, device)
                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                      (epoch + 1, valid_acc, valid_auc))
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                      (epoch + 1, test_acc, test_auc))
                if valid_acc > best_valid_accuracy:
                    best_valid_accuracy = valid_acc
                    best_valid_accuracy_1 = v_acc_1
                    best_valid_accuracy_2 = v_acc_2
                    best_test_accuracy_1 = t_acc_1
                    best_test_accuracy_2 = t_acc_2
                    best_test_accuracy = test_acc
                    torch.save(model.state_dict(), '%s/bestmodel.pth' % (modelsave_path))
                print('VALID BEST ACCURACY :%.3f AND TEST BEST ACCURACY: %.3f' %
                      (best_valid_accuracy, best_test_accuracy))
                print('验证集中最佳查准率 - 预测为1 且实际为1 ，看涨的准确率 :%.3f' %
                      best_valid_accuracy_1)
                print('验证集中最佳查全率 - 实际为1，预测为1 :%.3f' %
                      best_valid_accuracy_2)
                print('测试集中最佳查准率 - 预测为1 且实际为1 ，看涨的准确率 :%.3f' %
                      best_test_accuracy_1)
                print('测试集中最佳查全率 - 实际为1，预测为1 :%.3f' %
                      best_test_accuracy_2)
                print(f'loss:{running_losss}')
            model.train()
    model.eval()
    with torch.no_grad():
        latest_auc, latest_acc, l_acc_1, l_acc_2 = classification_scores_2(model, latestloader, device)
        print('LATEST ACCURACY: %.3f, LATEST AUROC: %.3f' %
              (latest_acc, latest_auc))
        print('最佳查准率 - 预测为1 且实际为1 ，看涨的准确率 :%.3f' %
              l_acc_1)
        print('最佳查全率 - 实际为1，预测为1 :%.3f' %
              l_acc_2)


if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger_conf.init_log('train_conv')
    DEVICE = torch.device(f"cuda:{opt.gpu_index}")
    print(f"Device is {DEVICE}.")
    modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, opt.run_name)
    os.makedirs(modelsave_path, exist_ok=True)
    train(DEVICE)
