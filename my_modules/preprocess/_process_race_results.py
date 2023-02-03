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
    
            #カテゴリ変数の処理
    def process_categorical(self):
        df_pe = self.data_pe.copy()
        le_horse = LabelEncoder().fit(self.data_pe['horse_id'])
        le_jockey = LabelEncoder().fit(self.data_pe['jockey_id'])
        le_trainer = LabelEncoder().fit(self.data_pe['trainer_id'])
        le_breeder = LabelEncoder().fit(self.data_pe['breeder_id'])
        self.le_horse = le_horse
        self.le_jockey = le_jockey
        self.le_trainer = le_trainer
        self.le_breeder = le_breeder
        
        #ラベルエンコーディング。horse_id, jockey_idを0始まりの整数に変換
        mask_horse = df['horse_id'].isin(le_horse.classes_)
        new_horse_id = df['horse_id'].mask(mask_horse).dropna().unique()
        le_horse.classes_ = np.concatenate([le_horse.classes_, new_horse_id])
        df['horse_id'] = le_horse.transform(df['horse_id'])
        
        
        mask_jockey = df['jockey_id'].isin(le_jockey.classes_)
        new_jockey_id = df['jockey_id'].mask(mask_jockey).dropna().unique()
        le_jockey.classes_ = np.concatenate([le_jockey.classes_, new_jockey_id])
        df['jockey_id'] = le_jockey.transform(df['jockey_id'])
        
        
        mask_trainer = df['trainer_id'].isin(le_trainer.classes_)
        new_trainer_id = df['trainer_id'].mask(mask_trainer).dropna().unique()
        le_trainer.classes_ = np.concatenate([le_trainer.classes_, new_trainer_id])
        df['trainer_id'] = le_trainer.transform(df['trainer_id'])
        
        
        mask_breeder = df['breeder_id'].isin(le_breeder.classes_)
        new_breeder_id = df['breeder_id'].mask(mask_breeder).dropna().unique()
        le_breeder.classes_ = np.concatenate([le_breeder.classes_, new_breeder_id])
        df['breeder_id'] = le_breeder.transform(df['breeder_id'])
        
        
        #horse_id, jockey_idをpandasのcategory型に変換
        df['horse_id'] = df['horse_id'].astype('category')
        df['jockey_id'] = df['jockey_id'].astype('category')
        df['trainer_id'] = df['trainer_id'].astype('category')
        df['breeder_id'] = df['breeder_id'].astype('category')
        
        
        #そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        #列を一定にするため
        weathers = df_pe['weather'].unique()
        race_types = df_pe['race_type'].unique()
        ground_states = df_pe['ground_state'].unique()
        sexes = df_pe['性'].unique()
        df['weather'] = pd.Categorical(df['weather'], weathers)
        df['race_type'] = pd.Categorical(df['race_type'], race_types)
        df['ground_state'] = pd.Categorical(df['ground_state'], ground_states)
        df['性'] = pd.Categorical(df['性'], sexes)
        df = pd.get_dummies(df, columns=['weather', 'race_type', 'ground_state', '性'])
        self.data_c = df    
