import numpy as np
import pandas as pd
from torch.utils.data import Dataset


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


def reformat_data(df):
    call_or_put = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, -3]
    df = df.iloc[:, 1:-3]
    _, d = df.shape
    day_0_data = df.iloc[:, :int(d / 5)]
    columns = day_0_data.columns
    no_need = ['PreClosePrice', 'PrePosition', 'RemainingTerm', 'PreSettlePrice', 'CallOrPut']
    for nn in no_need:
        columns = np.delete(columns, np.where(columns == nn))
    day_0_data = day_0_data[columns]
    day_1_data = df[columns + '_1'].copy()
    day_2_data = df[columns + '_2'].copy()
    day_3_data = df[columns + '_3'].copy()
    day_4_data = df[columns + '_4'].copy()

    day_0_data.loc[:, 'CallOrPut'] = call_or_put
    day_1_data.loc[:, 'CallOrPut'] = call_or_put
    day_2_data.loc[:, 'CallOrPut'] = call_or_put
    day_3_data.loc[:, 'CallOrPut'] = call_or_put
    day_4_data.loc[:, 'CallOrPut'] = call_or_put
    x = np.array([day_0_data.to_numpy(), day_1_data.to_numpy(), day_2_data.to_numpy(), day_3_data.to_numpy(),
                  day_4_data.to_numpy()])
    y = y.to_numpy().reshape(-1, 1)
    # x_0 = x[:, np.random.permutation(x.shape[1])]
    # x_0 = x_0[np.random.permutation(x.shape[0]), :]
    x = np.transpose(x, (1, 0, 2))
    # x_0 = np.transpose(x_0, (1, 0, 2))

    return x.astype(np.float32), y.astype(np.float32)


def data_prep_china_options(seed):
    """
    1。 分类数据转换成概率，变成连续型数据
    """
    np.random.seed(seed)
    PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/h_sh_300/'
    NORMAL_TYPE = 'mean_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv', parse_dates=['TradingDate'])
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv', parse_dates=['TradingDate'])
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv', parse_dates=['TradingDate'])
    cat_features = ['CallOrPut', 'MainSign']
    for i in range(1, 5):
        cat_features.append(f'MainSign_{i}')
    for cat in cat_features:
        for uni in training_df[cat].unique():
            p = training_df[training_df[cat] == uni].shape[0] / training_df.shape[0]
            training_df[cat].replace(to_replace=uni, value=p, inplace=True)
            validation_df[cat].replace(to_replace=uni, value=p, inplace=True)
            testing_df[cat].replace(to_replace=uni, value=p, inplace=True)
    # X_train, y_train = reformat_data(training_df)
    # X_valid, y_valid = reformat_data(validation_df)
    # X_test, y_test = reformat_data(testing_df)
    return training_df, validation_df, testing_df


class DataSetCatCon(Dataset):
    def __init__(self, data, trading_date_idxs=0):
        self.trading_dates = data.iloc[:, trading_date_idxs]
        self.unique_trading_dates = np.unique(self.trading_dates)
        # 删除时间那一列，因为时间不参数训练
        # X = np.delete(X,trading_date_idxs,1)
        self.data = data

    def __len__(self):
        return len(self.unique_trading_dates)

    def __getitem__(self, idx):
        trading_date = self.unique_trading_dates[idx]
        _idx = np.where(self.trading_dates == trading_date)
        _x = self.data.iloc[_idx].iloc[:, 1:]
        sub_x = _x.iloc[np.random.permutation(_x.shape[0])[:50], :]
        x, y = reformat_data(sub_x)
        # print(tmp_x.shape)
        x_tmp = sub_x.iloc[:, :-3]
        x_1 = x_tmp.to_numpy()[:, np.random.permutation(x_tmp.shape[1])]
        x_2 = x_tmp.to_numpy()
        return x.astype(np.float32), x_1.astype(np.float32), x_2.astype(np.float32), y.astype(np.float32)


if __name__ == '__main__':
    pass
