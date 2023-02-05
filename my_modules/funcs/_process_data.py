import pandas as pd

def change_answer_label(processed_df, label_type='bin'):
    """
    データを学習データと, 訓練データに分ける関数
    """
    if label_type=='bin':
        processed_df['rank'] = processed_df['rank'].map(lambda x:1 if x<4 else 0)
    elif label_type=='bias_top3':
        processed_df['rank'] = processed_df['rank'].map(lambda x: int(1/x * 10) if x<4 else 0)
    elif label_type=='bias_all':
        processed_df['rank'] = processed_df['rank'].map(lambda x: int(1/x * 10))
    elif label_type=='prize':
        # intじゃないとだめやねんて
        #  賞金期待値, 賞金そのものを予測させる
        # horse_results_processor.raw_data['賞金'] に 賞金データあり
        pass
        # processed_df['rank'] = processed_df['rank'].map(lambda x: 1/x )
    else:
        print('No such label_type')
        return None
    
    return processed_df



def make_data(processed_df,test_size=0.2):
    x_ = processed_df.drop(['rank','date','単勝'],axis=1)
    y_ = processed_df['rank']

    sorted_id_list = processed_df.sort_values("date").index.unique()
    train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
    x_train, x_test = x_.loc[train_id_list],x_.loc[test_id_list]
    y_train, y_test = y_.loc[train_id_list],y_.loc[test_id_list]

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