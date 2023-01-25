import numpy as np
import lightgbm as lgb


class LearnLGBM():
    

    def __init__(self,peds,results,horse_results):
        self.model = None
        self.model_ft = None
        self.date = '2022/12/31'
        self.pe = None
        self.r = None
        self.horse_results = None
        self.peds = peds
        self.results = results
        self.horse_results = horse_results
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.path_ft = '/Users/Owner/Desktop/Horse/horse/peds_ft.txt'

        self.lgbm_params = {
                'metric': 'ndcg',
                'objective': 'lambdarank',
                'ndcg_eval_at': [1,2,3],
                'boosting_type': 'gbdt',
                'random_state': 777,
                'lambdarank_truncation_level': 10,
                'learning_rate': 0.02273417953255777,
                'n_estimators': 97,
                'num_leaves': 42,
                'force_col_wise':True
            }



    def learn_model_ft(self,minn=2,maxn=14):
        path_ft = self.path_ft
        model_ft = ft.train_unsupervised(path_ft,dim=62,minn=minn,maxn=maxn)
        self.model_ft = model_ft


    def get_model_ft(self):
        return self.model_ft
    

    def process_pe(self,peds):
        pe = Peds(peds)
        pe.regularize_peds()
        # 血統データ　カテゴリ変数処理
        pe.categorize()
        # pe.vectorize(pe.peds_re,self.model_ft)
        self.pe = pe
        print("pe finish")
        print("pe regularizrd")


    def process_hr(self,results,horse_results):
        r = Results(results)
        r.preprocessing()
        #馬の過去成績データ追加
        hr = HorseResults(horse_results)
        self.hr = hr
        r.merge_horse_results(hr)
        r.merge_peds(self.pe.peds_cat)
        # r.merge_peds(pe.peds_cat)
        #カテゴリ変数の処理
        # pedsは既にカテゴリ化したdataをconcatしているので, ここでカテゴリ化せずとも良い
        r.process_categorical()
        self.r = r


    def process_data(self):
        peds = self.peds.copy()
        results = self.results.copy()
        horse_results = self.horse_results.copy()
        # self.learn_model_ft()
        self.process_pe(peds)
        self.process_hr(results,horse_results)
        
        
    def get_train_data(self,test_size=0.2,label_type='bin'):
        self.process_data()
        train, test = split_data(self.r.data_c.fillna(0),test_size=test_size,label_type=label_type)
        x_train = train.drop(['rank', 'date','単勝'], axis=1)
        y_train = train['rank']
        x_test = test.drop(['rank', 'date','単勝'], axis=1)
        y_test = test['rank']
        train_query = x_train.groupby(x_train.index).size()
        test_query = x_test.groupby(x_test.index).size()
        train = lgb.Dataset(x_train, y_train, group=train_query)
        test = lgb.Dataset(x_test, y_test, reference=train, group=test_query)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        return train, test

    
    def get_train_data2(self):
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        train_query = x_train.groupby(x_train.index).size()
        test_query = x_test.groupby(x_test.index).size()
        train = lgb.Dataset(x_train, y_train, group=train_query)
        test = lgb.Dataset(x_test, y_test, reference=train, group=test_query)
        return train, test


    def get_train_data3(featured_data,test_rate=1.0,is_rus=False,label_type='bin'):
        train,test = split_data(feature_enginnering.featured_data,label_type=label_type)
        x_train,y_train,_,_ = make_data(train,test_rate=test_rate,is_rus=False)
        x_test,y_test,_,_ = make_data(test,test_rate=test_rate,is_rus=False)

        train_query = x_train.groupby(x_train.index).size()
        test_query = x_test.groupby(x_test.index).size()
        train = lgb.Dataset(x_train, y_train, group=train_query)
        test = lgb.Dataset(x_test, y_test, reference=train, group=test_query)
        return train, test


    def learn_lgb(self,lgbm_params=None,test_size=0.2):
        if lgbm_params==None:
            lgbm_params = self.lgbm_params
        
        train, test = self.get_train_data(test_size=test_size)
        lgb_rank = lgb.train(
                lgbm_params,
                train,
                # valid_sets=test,
                num_boost_round=100,
                valid_names=['train'],
                # early_stopping_rounds=20,
            )

        self.model = lgb_rank

# train data を与えて学習させるのが learn_lgb2
    def learn_lgb2(self,train,lgbm_params=None):
        if lgbm_params==None:
            lgbm_params = self.lgbm_params
        
        lgb_rank = lgb.train(
                lgbm_params,
                train,
                # valid_sets=test,
                num_boost_round=100,
                valid_names=['train'],
                # early_stopping_rounds=20,
            )

        self.model = lgb_rank
    