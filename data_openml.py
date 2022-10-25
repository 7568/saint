import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset


def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text + ": {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487, 44, 1590, 42178, 1111, 31, 42733, 1494, 1017, 4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression': [541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]


def concat_data(X, y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:, 0].tolist(), columns=['target'])], axis=1)


def data_split(X, y, nan_mask, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise Exception('Shape of data not same as that of nan mask!')

    y_d = {
        'data': y[indices].reshape(-1, 1)
    }
    return x_d, y_d


def data_prep_openml(ds_id, seed, task, datasplit=[.65, .15, .2]):
    np.random.seed(seed)
    dataset = openml.datasets.get_dataset(ds_id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe",
                                                                    target=dataset.default_target_attribute)
    if ds_id == 42178:
        categorical_indicator = [True, False, True, True, False, True, True, True, True, True, True, True, True, True,
                                 True, True, True, False, False]
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp]
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        print(y.shape, X.shape)
    if ds_id in [42728, 42705, 42729, 42571]:
        # import ipdb; ipdb.set_trace()
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit, size=(X.shape[0],))

    train_indices = X[X.Set == "train"].index
    valid_indices = X[X.Set == "valid"].index
    test_indices = X[X.Set == "test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)

    cat_dims = []
    for col in categorical_columns:
        #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder()
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
        #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder()
        y = l_enc.fit_transform(y)
    X_train, y_train = data_split(X, y, nan_mask, train_indices)
    X_valid, y_valid = data_split(X, y, nan_mask, valid_indices)
    X_test, y_test = data_split(X, y, nan_mask, test_indices)

    train_mean, train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(0), np.array(
        X_train['data'][:, con_idxs], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std


def data_prep_china_options(seed):
    np.random.seed(seed)
    PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/h_sh_300/'
    NORMAL_TYPE = 'mean_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv',parse_dates=['TradingDate'])
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv',parse_dates=['TradingDate'])
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv',parse_dates=['TradingDate'])

    cat_features = ['CallOrPut', 'MainSign']
    cat_dims = [training_df[i].unique().size for i in cat_features]
    cat_idxs = [training_df.columns.get_loc(i) for i in cat_features]
    trading_date_idxs = training_df.columns.get_loc('TradingDate')
    con_idxs = np.delete(np.arange(training_df.columns.size - 1), np.concatenate((cat_idxs, [trading_date_idxs,-1,-2,-3])))
    X_train, y_train, X_valid, y_valid, X_test, y_test = {}, {}, {}, {}, {}, {}
    # X_train['data'] = training_df.iloc[:500, :-1].to_numpy()

    X_train['data'] = training_df.iloc[:, :-3].to_numpy()
    X_train['mask'] = np.ones(X_train['data'].shape)
    y_train['data'] = np.array(training_df['C_1']).reshape(-1, 1)
    X_valid['data'] = validation_df.iloc[:, :-3].to_numpy()
    X_valid['mask'] = np.ones(X_valid['data'].shape)
    y_valid['data'] = np.array(validation_df['C_1']).reshape(-1, 1)
    X_test['data'] = testing_df.iloc[:, :-3].to_numpy()
    X_test['mask'] = np.ones(X_test['data'].shape)
    y_test['data'] = np.array(testing_df['C_1']).reshape(-1, 1)

    # X_train['data'] = training_df.iloc[:500, :-2].to_numpy()
    # X_train['mask'] = np.ones(X_train['data'].shape)
    # y_train['data'] = np.array(training_df['target']).reshape(-1, 1)[:500, :]
    # X_valid['data'] = validation_df.iloc[:500, :-1].to_numpy()
    # X_valid['mask'] = np.ones(X_valid['data'].shape)
    # y_valid['data'] = np.array(validation_df['target']).reshape(-1, 1)[:500, :]
    # X_test['data'] = testing_df.iloc[:500, :-1].to_numpy()
    # X_test['mask'] = np.ones(X_test['data'].shape)
    # y_test['data'] = np.array(testing_df['target']).reshape(-1, 1)[:500, :]

    # train_mean, train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(0), np.array(
    #     X_train['data'][:, con_idxs], dtype=np.float32).std(0)
    # train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    train_mean, train_std=0,0
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, trading_date_idxs


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,con_cols, task='clf', continuous_mean_std=None,trading_date_idxs=0):
        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X = X['data'].copy()
        self.trading_dates = X[:,trading_date_idxs]
        self.unique_trading_dates = np.unique(self.trading_dates)
        # 删除时间那一列，因为时间不参数训练
        # X = np.delete(X,trading_date_idxs,1)

        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns
        if task == 'clf':
            self.y = Y['data']  # .astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std


    def __len__(self):
        return len(self.unique_trading_dates)

    def __getitem__(self, idx):
        trading_date = self.unique_trading_dates[idx]
        _idx = np.where(self.trading_dates == trading_date)

        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[_idx], self.X1[_idx]),axis=1), self.X2[_idx], self.y[_idx], np.concatenate(
            (self.cls_mask[_idx], self.X1_mask[_idx]),axis=1), self.X2_mask[_idx]
