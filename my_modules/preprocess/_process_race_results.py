import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class RaceResults():
    def __init__(self, results:pd.DataFrame):
        self.data = results
        self.le_peds = None
            
    #前処理    
    def preprocessing(self):
        df = self.data.copy()
        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)
#         rank学習の場合はそのまま
#         df['rank'] = df['着順'].map(lambda x:1 if x<4 else 0)
        df['rank'] = df['着順']
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
        df.drop(["タイム", "着差", "調教師", "性齢", "馬体重", '馬名', '騎手', '人気', '着順'],
                axis=1, inplace=True)
        df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")
        #開催場所
        df['開催'] = df.index.map(lambda x:str(x)[4:6])
        df['n_horse'] = df.index.map(lambda x: len(df.loc[x]))       
        self.data_p = df
        self.processed_df = df
    

    def get_processed_df(self):
        return self.processed_df