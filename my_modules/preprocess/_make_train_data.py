from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class MakeTrainData():
    
    
    def __init__(self,data_merger):
        self.data_merger = data_merger
        self.processed_df = self.data_merger.get_merged_df().copy()

    
    def get_processed_df(self):
        return self.processed_df.copy()
        

    def categorize_peds(self):
        merged_df = self.get_processed_df().copy()
        peds_processor = self.data_merger.peds_processor
        peds_processed = peds_processor.get_processed_df().copy()
        le_peds_dict = {}
        
        for column in peds_processed.columns:
            target_col = merged_df[column].astype(str).fillna('Na')
            le_peds_dict[column] = LabelEncoder().fit(target_col)
            merged_df[column] = le_peds_dict[column].transform(target_col)
            merged_df[column] = pd.Series(merged_df[column],dtype='category')
        
        print("--finish categorize peds--")
        self.le_peds_dict = le_peds_dict
        self.processed_df = merged_df
    
    
    # 血統データをベクトル化する関数, 未実装
    def vectorize_peds(self):
        pass


    def categorize_id(self):

        # horse_id, jockey_id, trainer_id, owner_id カテゴリ化
        merged_df = self.get_processed_df().copy()
        race_result_processor = self.data_merger.race_results_processor
        processed_df = race_result_processor.get_processed_df().copy()
        
        le_horse = LabelEncoder().fit(processed_df['horse_id'])
        le_jockey = LabelEncoder().fit(processed_df['jockey_id'])
        le_trainer = LabelEncoder().fit(processed_df['trainer_id'])
        le_owner = LabelEncoder().fit(processed_df['owner_id'])
        self.le_horse = le_horse
        self.le_jockey = le_jockey
        self.le_trainer = le_trainer
        self.le_owner = le_owner
        
        #ラベルエンコーディング。horse_id, jockey_idを0始まりの整数に変換
        mask_horse = merged_df['horse_id'].isin(le_horse.classes_)
        new_horse_id = merged_df['horse_id'].mask(mask_horse).dropna().unique()
        le_horse.classes_ = np.concatenate([le_horse.classes_, new_horse_id])
        merged_df['horse_id'] = le_horse.transform(merged_df['horse_id'])
        
        
        mask_jockey = merged_df['jockey_id'].isin(le_jockey.classes_)
        new_jockey_id = merged_df['jockey_id'].mask(mask_jockey).dropna().unique()
        le_jockey.classes_ = np.concatenate([le_jockey.classes_, new_jockey_id])
        merged_df['jockey_id'] = le_jockey.transform(merged_df['jockey_id'])
        
        
        mask_trainer = merged_df['trainer_id'].isin(le_trainer.classes_)
        new_trainer_id = merged_df['trainer_id'].mask(mask_trainer).dropna().unique()
        le_trainer.classes_ = np.concatenate([le_trainer.classes_, new_trainer_id])
        merged_df['trainer_id'] = le_trainer.transform(merged_df['trainer_id'])
        
        
        mask_owner = merged_df['owner_id'].isin(le_owner.classes_)
        new_owner_id = merged_df['owner_id'].mask(mask_owner).dropna().unique()
        le_owner.classes_ = np.concatenate([le_owner.classes_, new_owner_id])
        merged_df['owner_id'] = le_owner.transform(merged_df['owner_id'])
        
        
        #horse_id, jockey_idをpandasのcategory型に変換
        merged_df['horse_id'] = merged_df['horse_id'].astype('category')
        merged_df['jockey_id'] = merged_df['jockey_id'].astype('category')
        merged_df['trainer_id'] = merged_df['trainer_id'].astype('category')
        merged_df['owner_id'] = merged_df['owner_id'].astype('category')
        print("--finish categorize id--")

        self.processed_df = merged_df
        
        
    def dumminize(self):
        #そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        #列を一定にするため
        merged_df = self.get_processed_df().copy()
        weathers = merged_df['weather'].unique()
        race_types = merged_df['race_type'].unique()
        ground_states = merged_df['ground_state'].unique()
        sexes = merged_df['性'].unique()
        places = merged_df['開催'].unique()
        merged_df['weather'] = pd.Categorical(merged_df['weather'], weathers)
        merged_df['race_type'] = pd.Categorical(merged_df['race_type'], race_types)
        merged_df['ground_state'] = pd.Categorical(merged_df['ground_state'], ground_states)
        merged_df['性'] = pd.Categorical(merged_df['性'], sexes)
        merged_df['開催'] = pd.Categorical(merged_df['開催'], places)
        dumminized_df = pd.get_dummies(merged_df, columns=['weather', 'race_type', 'ground_state', '性', '開催'])
        print("--finish dumminize--")

        self.processed_df = dumminized_df  

    
    def add_date_cosin(self):
        """
        日付を角度に変換する
        うるう年の影響は十分小さいものとする
        """
        month_dict = {
            1: 0,
            2: 31,
            3: 59,
            4: 90,
            5: 120,
            6: 151,
            7: 181,
            8: 212,
            9: 243,
            10: 273,
            11: 304,
            12: 334
        }
        merged_df = self.data_merger.get_merged_df().copy()
        date_row = pd.to_datetime(merged_df['date'])
        date_row = date_row.map(lambda x : month_dict[x.month]+x.day)
        theta = np.radians(360*(date_row/365))
        print("--finish add cosin--")

        self.processed_df['date_cosin'] = np.cos(theta)


    