import torch
from sklearn.metrics import roc_auc_score, mean_squared_error
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


# def classification_scores(model, dloader, device, task, vision_dset):
#     model.eval()
#     m = nn.Softmax(dim=1)
#     y_test = torch.empty(0).to(device)
#     y_pred = torch.empty(0).to(device)
#     prob = torch.empty(0).to(device)
#     with torch.no_grad():
#         for i, data in enumerate(dloader, 0):
#             x_categ, x_cont, y_gts, cat_mask, con_mask = data[0], data[1], data[2], data[3], data[4]
#             x_categ, x_cont, y_gts, cat_mask, con_mask = torch.squeeze(x_categ, 0).to(device), \
#                                                          torch.squeeze(x_cont, 0).to(device), \
#                                                          torch.squeeze(y_gts, 0).to(device), \
#                                                          torch.squeeze(cat_mask, 0).to(device), \
#                                                          torch.squeeze(con_mask, 0).to(device)
#             _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
#             reps = model.transformer(x_categ_enc, x_cont_enc)
#             y_reps = reps[:, 0, :]
#             y_outs = model.mlpfory(y_reps)
#             # import ipdb; ipdb.set_trace()
#             y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
#             y_pred = torch.cat([y_pred, torch.argmax(y_outs, dim=1).float()], dim=0)
#             if task == 'binary':
#                 prob = torch.cat([prob, m(y_outs)[:, -1].float()], dim=0)
#
#     correct_results_sum = (y_pred == y_test).sum().float()
#     acc = correct_results_sum / y_test.shape[0] * 100
#     auc = 0
#     if task == 'binary':
#         auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
#     return acc.cpu().numpy(), auc


def mean_sq_error(model, dloader, device):
    model.eval()
    with torch.no_grad():
        y = []
        y_hat = []
        for i, data in tqdm(enumerate(dloader, 0), total=len(dloader)):
            x, x_1, x_2, _y = data[0], data[1], data[2], data[3]
            x, x_1, x_2, _y = torch.squeeze(x, 0).to(device), torch.squeeze(x_1, 0).to(device), torch.squeeze(x_2, 0).to(
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
