from typing import Tuple

import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm

from augmentations import embed_data_mask
import torch.nn as nn


def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:, -1] = 0
    return mask


def tag_gen(tag, y):
    return np.repeat(tag, len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[args.epochs // 2.667, args.epochs // 1.6,
                                                                     args.epochs // 1.142], gamma=0.1)
    return scheduler


# def imputations_acc_justy(model, dloader, device):
#     model.eval()
#     m = nn.Softmax(dim=1)
#     y_test = torch.empty(0).to(device)
#     y_pred = torch.empty(0).to(device)
#     prob = torch.empty(0).to(device)
#     with torch.no_grad():
#         for i, data in enumerate(dloader, 0):
#             x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[
#                 3].to(device)
#             _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
#             reps = model.transformer(x_categ_enc, x_cont_enc)
#             y_reps = reps[:, model.num_categories - 1, :]
#             y_outs = model.mlpfory(y_reps)
#             # import ipdb; ipdb.set_trace()
#             y_test = torch.cat([y_test, x_categ[:, -1].float()], dim=0)
#             y_pred = torch.cat([y_pred, torch.argmax(m(y_outs), dim=1).float()], dim=0)
#             prob = torch.cat([prob, m(y_outs)[:, -1].float()], dim=0)
#
#     correct_results_sum = (y_pred == y_test).sum().float()
#     acc = correct_results_sum / y_test.shape[0] * 100
#     auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
#     return acc, auc


# def multiclass_acc_justy(model, dloader, device):
#     model.eval()
#     vision_dset = True
#     m = nn.Softmax(dim=1)
#     y_test = torch.empty(0).to(device)
#     y_pred = torch.empty(0).to(device)
#     prob = torch.empty(0).to(device)
#     with torch.no_grad():
#         for i, data in enumerate(dloader, 0):
#             x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[
#                 3].to(device)
#             _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
#             reps = model.transformer(x_categ_enc, x_cont_enc)
#             y_reps = reps[:, model.num_categories - 1, :]
#             y_outs = model.mlpfory(y_reps)
#             # import ipdb; ipdb.set_trace()
#             y_test = torch.cat([y_test, x_categ[:, -1].float()], dim=0)
#             y_pred = torch.cat([y_pred, torch.argmax(m(y_outs), dim=1).float()], dim=0)
#
#     correct_results_sum = (y_pred == y_test).sum().float()
#     acc = correct_results_sum / y_test.shape[0] * 100
#     return acc, 0


def classification_scores(model, dloader, device):
    model.eval()
    m = nn.Softmax(dim=1)
    y = []
    y_hat = torch.empty(0).to(device)
    y_prob_hat = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dloader, 0), total=len(dloader)):
            x, x_1, x_2, _y = data[0], data[1], data[2], data[3]
            x, x_1, x_2, _y = torch.squeeze(x, 0).to(device), torch.squeeze(x_1, 0).to(device), torch.squeeze(x_2,
                                                                                                              0).to(
                device), torch.squeeze(_y, 0).to(device)
            y = np.append(y, _y.cpu().numpy())
            feature_out_0, predict_out = model(x, x_2)
            y_hat = torch.cat([y_hat, torch.argmax(predict_out, dim=1).float()], dim=0)
            y_prob_hat = torch.cat([y_prob_hat, m(predict_out)[:, -1].float()], dim=0)
    return eval_result(y_prob_hat,y_hat,y)

def eval_result(y_prob_hat,y_hat,y):
    auc = roc_auc_score(y_true=y, y_score=y_prob_hat.cpu())
    accu = accuracy_score(y_true=y, y_pred=y_hat.cpu())
    y_true = y
    y_test_hat = y_hat.cpu()
    tn, fp, fn, tp = confusion_matrix(y_true, y_test_hat).ravel()
    print('0：不涨 ， 1：涨')
    print('tn, fp, fn, tp', tn, fp, fn, tp)

    print(f'test中为1的比例 : {y_true.sum() / len(y_true)}')
    print(f'test中为0的比例 : {(1 - y_true).sum() / len(y_true)}')

    # error_in_test = mean_squared_error(y_test_hat, np.array(testing_df[target_fea]).reshape(-1, 1))
    accu_1 = tp / (tp + fp)
    accu_2 = tp / (tp + fn)
    print(f'查准率 - 预测为1 且实际为1 ，看涨的准确率: {accu_1}')
    print(f'查全率 - 实际为1，预测为1 : {accu_2}')
    print(f'F1 = {(2 * tp) / (len(y_true) + tp - tn)}')

    # print(f'AUC：{auc(y_true,y_test_hat)}')
    print(f'总体准确率：{accuracy_score(y_true, y_test_hat)}')
    return auc, accu, accu_1, accu_2

def classification_scores_2(model, dloader, device) -> Tuple[float, float, float, float]:
    model.eval()
    m = nn.Softmax(dim=1)
    y = []
    y_hat = torch.empty(0).to(device)
    y_prob_hat = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dloader, 0), total=len(dloader)):
            x, _y = data[0].to(device), data[1].to(device)

            y = np.append(y, _y.cpu().numpy())
            predict_out = model(x)
            y_hat = torch.cat([y_hat, torch.argmax(predict_out, dim=1).float()], dim=0)
            y_prob_hat = torch.cat([y_prob_hat, m(predict_out)[:, -1].float()], dim=0)
    return eval_result(y_prob_hat,y_hat,y)


def mean_sq_error(model, dloader, device):
    model.eval()
    with torch.no_grad():
        y = []
        y_hat = []
        for i, data in tqdm(enumerate(dloader, 0), total=len(dloader)):
            x, x_1, x_2, _y = data[0], data[1], data[2], data[3]
            x, x_1, x_2, _y = torch.squeeze(x, 0).to(device), torch.squeeze(x_1, 0).to(device), torch.squeeze(x_2,
                                                                                                              0).to(
                device), torch.squeeze(_y, 0).to(device)
            y = np.append(y, _y.cpu().numpy())
            feature_out_0, predict_out = model(x, x_2)
            y_hat = np.append(y_hat, predict_out.cpu().numpy())
        # import ipdb; ipdb.set_trace() 
        # rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        # print(np.array(y).shape)
        # print(np.array(y_hat).shape)
        rmse = mean_squared_error(y, y_hat)
        return rmse
