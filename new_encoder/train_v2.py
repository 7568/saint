import argparse
import os
import sys
from multiprocessing import cpu_count

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot
from new_encoder.new_model import NewModel

from new_encoder.data_openml import data_prep_china_options, DataSetCatCon
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from new_encoder.utils import count_parameters, classification_scores
from augmentations import embed_data_mask
from augmentations import add_noise
import os
import numpy as np
import logger_conf

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_index', default=6, type=int)
parser.add_argument('--expand_size', default=1, type=int)
parser.add_argument('--vision_dset', action='store_true')
parser.add_argument('--task', default='binary', type=str, choices=['binary', 'multiclass', 'regression'])
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
opt.pretrain = True
modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'

# device = torch.device(f"cuda:{opt.gpu_index}" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")

device = torch.device(f"cuda:{opt.gpu_index}")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
model = NewModel(28, opt.expand_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)

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

training_df, validation_df, testing_df = data_prep_china_options(opt.dset_seed)

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

print(opt)

print(f'cpu_count() : {cpu_count()}')
train_ds = DataSetCatCon(training_df)
# trainloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=int(cpu_count() * 0.7))
trainloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

valid_ds = DataSetCatCon(validation_df)
validloader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=4)

test_ds = DataSetCatCon(testing_df)
testloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

# model = NewModel(
#     categories=tuple(cat_dims),
#     num_continuous=len(con_idxs),
#     dim=opt.embedding_size,
#     dim_out=1,
#     depth=opt.transformer_depth,
#     heads=opt.attention_heads,
#     attn_dropout=opt.attention_dropout,
#     ff_dropout=opt.ff_dropout,
#     mlp_hidden_mults=(4, 2),
#     cont_embeddings=opt.cont_embeddings,
#     attentiontype=opt.attentiontype,
#     final_mlp_style=opt.final_mlp_style,
#     y_dim=y_dim
# )


# print(count_parameters(model))
# import ipdb; ipdb.set_trace()
# model.cuda()

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
delta = 0.5
print('Training begins now.')
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        optimizer.zero_grad()
        x, x_1, x_2, y = data[0], data[1], data[2], data[3]
        x, x_1, x_2, y = torch.squeeze(x, 0).to(device), torch.squeeze(x_1, 0).to(device), torch.squeeze(x_2, 0).to(
            device), torch.squeeze(y, 0).long().to(device)
        # print(x.shape)

        feature_out, predict_out = model(x, x_2)
        # print(predict_out)
        # print(y.squeeze())

        loss = criterion(predict_out, y.squeeze())
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
        # print(delta*loss_0.item())
        # print(loss_1.item())
    # print(running_loss)
    if opt.active_log:
        logger_conf.logger.info({'epoch': epoch, 'train_epoch_loss': running_loss, 'loss': loss.item()})
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():

            valid_acc,valid_auc = classification_scores(model, validloader, device)
            test_acc,test_auc = classification_scores(model, testloader, device)
            print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                  (epoch + 1, valid_acc, valid_auc))
            print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                  (epoch + 1, test_acc, test_auc))
            if valid_acc > best_valid_accuracy:
                best_valid_accuracy = valid_acc
                # if auroc > best_valid_auroc:
                #     best_valid_auroc = auroc
                best_test_auroc = test_auc
                best_test_accuracy = test_acc
                torch.save(model.state_dict(), '%s/bestmodel.pth' % (modelsave_path))
        model.train()

total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' % (total_parameters))
print('AUROC on best model:  %.3f' % (best_test_auroc))
if opt.active_log:
    logger_conf.logger.info({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc ,
        'test_accuracy_bestep':best_test_accuracy })
