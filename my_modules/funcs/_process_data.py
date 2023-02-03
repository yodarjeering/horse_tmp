import pandas as pd


def split_data(df, test_size=0.2, label_type='bin'):
    """
    データを学習データと, 訓練データに分ける関数
    """
    df_ = df.copy()
    if label_type=='bin':
        df_['rank'] = df_['rank'].map(lambda x:1 if x<4 else 0)
    elif label_type=='bias_top3':
        df_['rank'] = df_['rank'].map(lambda x: int(1/x * 10) if x<4 else 0)
    elif label_type=='bias_all':
        df_['rank'] = df_['rank'].map(lambda x: int(1/x * 10))
    elif label_type=='rank':
        # intじゃないとだめやねんて
        pass
        # df_['rank'] = df_['rank'].map(lambda x: 1/x )
    else:
        print('No such label_type')
        return None,None

    sorted_id_list = df_.sort_values("date").index.unique()
    train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
    train = df_.loc[train_id_list]#.drop(['date'], axis=1)
    test = df_.loc[test_id_list]#.drop(['date'], axis=1)
    return train, test

def make_data(data_,test_rate=0.8,is_rus=True):
    x_ = data_.drop(['rank','date','単勝'],axis=1)
    y_ = data_['rank']

    test_rate = int(test_rate*len(x_))
    x_train, x_test = x_.iloc[:test_rate],x_.iloc[test_rate:]
    y_train, y_test = y_.iloc[:test_rate],y_.iloc[test_rate:]
    if is_rus:
        rus = RandomUnderSampler(random_state=0)
        x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
        return x_resampled, y_resampled, x_test, y_test
    else:
        return x_train,y_train,x_test,y_test

def update_data(old, new):
    """
    Parameters:
    ----------
    old : pandas.DataFrame
        古いデータ
    new : pandas.DataFrame
        新しいデータ
    """

    filtered_old = old[~old.index.isin(new.index)]
    return pd.concat([filtered_old, new])


