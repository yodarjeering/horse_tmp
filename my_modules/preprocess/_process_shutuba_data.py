import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class ShutubaData():
    def __init__(self, shutuba_data, race_results):
        self.shutuba_data = shutuba_data
        self.race_results = race_results
            
    #前処理    
    def preprocessing(self):
        df = self.shutuba_data.copy()
        # 性齢を性と年齢に分ける
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)
        # 馬体重を体重と体重変化に分ける
        df["体重"] = df["馬体重"].str.split("(", expand=True)[0].astype(int)
        df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)
        # データをint, floatに変換
        df["単勝"] = df["単勝"].astype(float)
        df["course_len"] = df["course_len"].astype(float) // 100
        # 不要な列を削除
        df.drop(["性齢", "馬体重",  "人気", "around", "race_class"],
                axis=1, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        #開催場所
        df['開催'] = df.index.map(lambda x:str(x)[4:6])
        df['n_horse'] = df.index.map(lambda x: len(df.loc[x]))  
        
        # shutuba_data にowner_id がないため, race_results テーブルからowner_id 引っ張ってくる
        # df['owner_id'] = self.add_owner_id()    
        owner_id_list = self.add_owner_id()
        df['owner_id'] = owner_id_list

        df['枠番'] = df['枠番'].astype(int)
        df['馬番'] = df['馬番'].astype(int)
        df['斤量'] = df['斤量'].astype(float)
        df['horse_id'] = df['horse_id'].astype(int)
        df['jockey_id'] = df['jockey_id'].astype(int)
        df['trainer_id'] = df['trainer_id'].astype(int)
        df['owner_id'] = df['owner_id'].astype(int)

        self.data_p = df
        self.processed_df = df
    

    def add_owner_id(self):
        shutuba_data = self.shutuba_data.copy()
        race_results = self.race_results.copy()
        shutuba_data['horse_id'] = shutuba_data['horse_id'].astype(int)
        horse_id_list = shutuba_data['horse_id'].tolist() # <= int に直す必要あり
        owner_id_list = []
        
        for horse_id in horse_id_list:
            owner_id = race_results[race_results['horse_id']==horse_id]['owner_id'].iloc[0]
            owner_id_list.append(owner_id)

        return owner_id_list

    def get_processed_df(self):
        return self.processed_df