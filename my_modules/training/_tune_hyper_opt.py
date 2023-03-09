from hyperopt import hp, tpe,STATUS_OK
from hyperopt.fmin import fmin
import numpy as np
import lightgbm as lgb


# rank_xendcg 計算が早く挙動はlambdarankと似ている。
# ラベルはint型である必要があり, 大きい数字にはより良い意味を持たせる必要がある。 (例. 0:悪い, 1:普通, 2:良い, 3:かなり良い)

def optimize(trials, train_set, valid_set):
#探索スペース
    space = {
        #-----------------定数
        'objective': 'lambdarank', 
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'random_state': 777,
        'verbosity':-1, # マイナスで, 非表示
        'ndcg_eval_at': [1,2,3],# 上位3着を考慮する
        'force_col_wise':True,
        
        #--------------探索範囲
        'reg_alpha' : hp.loguniform('reg_alpha', np.log(1e-3), np.log(1e+1)),
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-3), np.log(1e-1)),
        'subsample' : hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1.0),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-2), np.log(1e+3)),
        
        #------------intに直す必要がある
        'lambdarank_truncation_level': hp.quniform('lambdarank_truncation_level',1,20,2), # ラムダの計算, 20レコードまで計算する
        'subsample_freq': hp.quniform('subsample_freq', 1, 20, 2),
        'min_child_samples': hp.quniform('min_child_samples', 1, 50, 2),
        # 'num_leaves': hp.quniform('num_leaves', 4, 100, 4), <= なぜかnum_leaves あると list index out エラー
    }

    max_evals = 25      #探索回数(25くらいで十分)
    
    # 内部関数
    def score(params):
        print("Training start:")
        lgb_results={}  #履歴格納用
        
        #----------------- quniform 型は, int に直す必要あり
        params['lambdarank_truncation_level'] = int(params['lambdarank_truncation_level'])
        params['subsample_freq'] = int(params['subsample_freq'])
        params['min_child_samples'] = int(params['min_child_samples'])
        # params['num_leaves'] = int(params['num_leaves'])

        lgb_clf = lgb.train(
            params,
            train_set,
            valid_sets=valid_set,
            valid_names=['valid'],
            early_stopping_rounds=100,
            evals_result=lgb_results,
        )
        return {'loss': -1.0 * lgb_results['valid']['ndcg@3'][lgb_clf.best_iteration], 'status': STATUS_OK}

    best_params = fmin(
            score,
            space,
            algo=tpe.suggest,
            trials=trials, 
            max_evals=max_evals,
            rstate=np.random.default_rng(0)
            )

    print("best parameters:", best_params)
    return best_params