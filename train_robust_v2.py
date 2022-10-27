import argparse
import os
from multiprocessing import cpu_count

import numpy as np
import torch
from torch import nn
from models import SAINT, SAINT_vision

from data_openml import data_prep_china_options, DataSetCatCon
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from utils import count_parameters, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise

import os
import numpy as np
import logger_conf

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_index', default=6, type=int)
parser.add_argument('--vision_dset', action='store_true')
parser.add_argument('--task', default='regression', type=str, choices=['binary', 'multiclass', 'regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,
                    choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=1, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default=1, type=int)
parser.add_argument('--dset_seed', default=1, type=int)
parser.add_argument('--active_log', action='store_true')

parser.add_argument('--log_to_file', action='store_true')

parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*',
                    choices=['contrastive', 'contrastive_sim', 'denoising'])
parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_noise_type', default=None, type=str, choices=['missing', 'cutmix'])
parser.add_argument('--train_noise_level', default=0, type=float)

parser.add_argument('--ssl_samples', default=None, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])

opt = parser.parse_args()
if opt.log_to_file:
    logger_conf.init_log('train_robust_v2')
opt.pretrain=True
modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'

device = torch.device(f"cuda:{opt.gpu_index}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:

    if opt.train_noise_type is not None and opt.train_noise_level > 0:
        logger_conf.logger.info("saint_v2_robustness", f'{opt.run_name}_{opt.task}',
                   f'{opt.task}_{opt.train_noise_type}_{str(opt.train_noise_level)}_{str(opt.attentiontype)}_{str(opt.dset_id)}')
    elif opt.ssl_samples is not None:
        logger_conf.logger.info("saint_v2_ssl", f'{opt.run_name}_{opt.task}',
                   f'{opt.task}_{str(opt.ssl_samples)}_{str(opt.attentiontype)}_{str(opt.dset_id)}')
    else:
        raise Exception('wrong config.check the file you are running')

cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, \
trading_date_idxs = data_prep_china_options(opt.dset_seed)
# continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)
continuous_mean_std = None

##### Setting some hyperparams based on inputs and dataset
_, nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(4, opt.embedding_size)
    # opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = 4
    opt.attention_dropout = 0.8
    opt.embedding_size = 16
    if opt.optimizer == 'SGD':
        opt.ff_dropout = 0.4
        opt.lr = 0.01
    else:
        opt.ff_dropout = 0.8

print(nfeat, opt.batchsize)
print(opt)

print(f'cpu_count() : {cpu_count()}')
train_ds = DataSetCatCon(X_train, y_train, cat_idxs, con_idxs, opt.dtask, continuous_mean_std, trading_date_idxs)
trainloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=int(cpu_count() * 0.7))

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, con_idxs, opt.dtask, continuous_mean_std, trading_date_idxs)
validloader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs, con_idxs, opt.dtask, continuous_mean_std, trading_date_idxs)
testloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:, 0]))

cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(
    int)  # Appending 1 for CLS token, this is later used to generate embeddings.

model = SAINT(
    categories=tuple(cat_dims),
    num_continuous=len(con_idxs),
    dim=opt.embedding_size,
    dim_out=1,
    depth=opt.transformer_depth,
    heads=opt.attention_heads,
    attn_dropout=opt.attention_dropout,
    ff_dropout=opt.ff_dropout,
    mlp_hidden_mults=(4, 2),
    cont_embeddings=opt.cont_embeddings,
    attentiontype=opt.attentiontype,
    final_mlp_style=opt.final_mlp_style,
    y_dim=y_dim
)
vision_dset = opt.vision_dset

if y_dim == 2 and opt.task == 'binary':
    # opt.task = 'binary'
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and opt.task == 'multiclass':
    # opt.task = 'multiclass'
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise Exception('case not written yet')

# print(count_parameters(model))
# import ipdb; ipdb.set_trace()
model.to(device)

if opt.pretrain:
    from pretraining import SAINT_pretrain

    model = SAINT_pretrain(model, cat_idxs,con_idxs, X_train, y_train, continuous_mean_std, opt, device,trading_date_idxs)

if opt.ssl_samples is not None and opt.ssl_samples > 0:
    print('We are in semi-supervised learning case')
    train_pts_touse = np.random.choice(X_train['data'].shape[0], opt.ssl_samples)
    X_train['data'] = X_train['data'][train_pts_touse, :]
    y_train['data'] = y_train['data'][train_pts_touse]

    X_train['mask'] = X_train['mask'][train_pts_touse, :]
    train_bsize = min(opt.ssl_samples // 4, opt.batchsize)

    train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std,trading_date_idxs)
    trainloader = DataLoader(train_ds, batch_size=train_bsize, shuffle=True, num_workers=4)

## Choosing the optimizer

if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler

    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000
print('Training begins now.')
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        optimizer.zero_grad()
        # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont. 
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0], data[1], data[2], data[3], data[4]
        x_categ, x_cont, y_gts, cat_mask, con_mask = torch.squeeze(x_categ, 0).to(device), \
                                                     torch.squeeze(x_cont, 0).to(device), \
                                                     torch.squeeze(y_gts, 0).to(device), \
                                                     torch.squeeze(cat_mask, 0).to(device), \
                                                     torch.squeeze(con_mask, 0).to(device)

        if opt.train_noise_type is not None and opt.train_noise_level > 0:
            noise_dict = {
                'noise_type': opt.train_noise_type,
                'lambda': opt.train_noise_level
            }
            if opt.train_noise_type == 'cutmix':
                x_categ, x_cont = add_noise(x_categ, x_cont, noise_params=noise_dict)
            elif opt.train_noise_type == 'missing':
                cat_mask, con_mask = add_noise(cat_mask, con_mask, noise_params=noise_dict)
        # We are converting the data to embeddings in the next step
        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
        reps = model.transformer(x_categ_enc, x_cont_enc)
        # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:, 0, :]

        y_outs = model.mlpfory(y_reps)
        if opt.task == 'regression':
            loss = criterion(y_outs, y_gts)
        else:
            loss = criterion(y_outs, y_gts.squeeze())
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    # print(running_loss)
    if opt.active_log:
        logger_conf.logger.info({'epoch': epoch, 'train_epoch_loss': running_loss,
                   'loss': loss.item()
                   })
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():

            valid_rmse = mean_sq_error(model, validloader, device, vision_dset)
            test_rmse = mean_sq_error(model, testloader, device, vision_dset)
            print('[EPOCH %d] VALID RMSE: %.3f' %
                  (epoch + 1, valid_rmse))
            print('[EPOCH %d] TEST RMSE: %.3f' %
                  (epoch + 1, test_rmse))
            if opt.active_log:
                logger_conf.logger.info({'valid_rmse': valid_rmse, 'test_rmse': test_rmse})
            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                best_test_rmse = test_rmse
                torch.save(model.state_dict(), '%s/bestmodel.pth' % (modelsave_path))
        model.train()

total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' % (total_parameters))
if opt.task == 'binary':
    print('AUROC on best model:  %.3f' % (best_test_auroc))
elif opt.task == 'multiclass':
    print('Accuracy on best model:  %.3f' % (best_test_accuracy))
else:
    print('RMSE on best model:  %.3f' % (best_test_rmse))

if opt.active_log:
    if opt.task == 'regression':
        logger_conf.logger.info({'total_parameters': total_parameters, 'test_rmse_bestep': best_test_rmse,
                   'cat_dims': len(cat_idxs), 'con_dims': len(con_idxs)})
    else:
        logger_conf.logger.info({'total_parameters': total_parameters, 'test_auroc_bestep': best_test_auroc,
                   'test_accuracy_bestep': best_test_accuracy, 'cat_dims': len(cat_idxs), 'con_dims': len(con_idxs)})
