import numpy as np

from data import data_prep,DataSetCatCon
import xgboost as xgb
from vime_utils import convert_matrix_to_vector, convert_vector_to_matrix
import argparse
from vime_utils import perf_metric


def init_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='forest', type=str,
                        choices=['1995_income', 'bank_marketing', 'qsar_bio', 'online_shoppers', 'blastchar', 'htru2',
                                 'shrutime', 'spambase', 'philippine', 'mnist', 'loan_data', 'arcene', 'volkert',
                                 'creditcard', 'arrhythmia', 'forest', 'kdd99'])
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str,
                        choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batchsize', default=1024, type=int)
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=4, type=int)
    parser.add_argument('--active_log', action='store_true')

    parser.add_argument('--pretrain', default=False, action='store_true')
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*',
                        choices=['contrastive', 'contrastive_sim', 'denoising'])
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix', 'gauss_noise'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)

    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)

    parser.add_argument('--ssl_avail_y', default=0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)

    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])

    return parser.parse_args()


def xgb_model(x_train, y_train,x_vaidation, y_validation, x_test):
    """XGBoost.

    Args:
      - x_train, y_train: training dataset
      - x_test: testing feature

    Returns:
      - y_test_hat: predicted values for x_test
    """
    # Convert labels into proper format
    # if len(y_train.shape) > 1:
    #     y_train = convert_matrix_to_vector(y_train)

        # Define and fit model on training dataset
        # - max_depth = 8,
        # - learning_rate = 0.01,
        # - tree_method = 'hist',
        # - subsample = 0.75,
        # - colsample_bytree = 0.75,
        # - reg_alpha = 0.5,
        # - reg_lambda = 0.5,
    params = {
        'max_depth': 8,
        'learning_rate': 0.01,
        'tree_method': 'hist',
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'n_estimators' : 50000,

    }
    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train,eval_set=[(x_vaidation, y_validation)],early_stopping_rounds=20,eval_metric=['auc'])

    # Predict on x_test
    y_test_hat = model.predict_proba(x_test)

    return y_test_hat

if __name__ == '__main__':

    opt = init_params()
    mask_params = {
        "mask_prob": opt.train_mask_prob,
        "avail_train_y": 0,
        "test_mask": opt.train_mask_prob
    }
    metric = 'auc'
    cat_dims, cat_idxs, con_idxs, X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, X_TEST, Y_TEST, train_mean, train_std = data_prep(opt.dataset, opt.set_seed)

    continuous_mean_std = None
    continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32)
    if continuous_mean_std is not None:
        mean, std = continuous_mean_std
        X_TRAIN_X1 = X_TRAIN['data'][:, cat_idxs].copy().astype(np.int64)  # categorical columns
        X_TRAIN_X2 = X_TRAIN['data'][:, con_idxs].copy().astype(np.float32)  # numerical columns
        X_TRAIN_X2= (X_TRAIN_X2 - mean) / std

        X_VALID_X1 = X_VALID['data'][:, cat_idxs].copy().astype(np.int64)  # categorical columns
        X_VALID_X2 = X_VALID['data'][:, con_idxs].copy().astype(np.float32)  # numerical columns
        X_VALID_X2 = (X_VALID_X2 - mean) / std

        X_TEST_X1 = X_TEST['data'][:, cat_idxs].copy().astype(np.int64)  # categorical columns
        X_TEST_X2 = X_TEST['data'][:, con_idxs].copy().astype(np.float32)  # numerical columns
        X_TEST_X2 = (X_TEST_X2 - mean) / std
        X_TRAIN['data'] = np.concatenate((X_TRAIN_X2, X_TRAIN_X1),axis=1)
        X_VALID['data'] = np.concatenate((X_VALID_X2, X_VALID_X1),axis=1)
        X_TEST['data'] = np.concatenate((X_TEST_X2, X_TEST_X1),axis=1)

    y_test_hat = xgb_model(X_TRAIN['data'], Y_TRAIN['data'], X_VALID['data'],Y_VALID['data'],X_TEST['data'])
    performance = perf_metric(metric, Y_TEST['data'], y_test_hat)
    print('auc : ',performance)
    metric = 'acc'
    performance = perf_metric(metric, Y_TEST['data'], y_test_hat)
    print('acc : ',performance)
