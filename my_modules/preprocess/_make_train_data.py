from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class MakeTrainData():
    
    
    def __init__(self,data_merger):
        self.data_merger = data_merger
        self.processed_df : pd.DataFrame = None
        

    def categorize_peds(self):
        merged_df = self.data_merger.get_merged_df()
        peds_processor = self.data_merger.peds_processor
        peds_processed = peds_processor.processed_df
        le_peds_dict = {}
        
        for column in peds_processed.columns:
            le_peds_dict[column] = LabelEncoder()
            merged_df[column] = le_peds_dict[column].fit_transform(peds_processed[column].fillna('Na'))
        
        print("finish categorize peds")
        self.le_peds_dict = le_peds_dict
        self.processed_df = merged_df
    
    
    # 血統データをベクトル化する関数, 未実装
    def vectorize_peds(self):
        pass


    def categorized_id(self):

        # horse_id, jockey_id, trainer_id, breeder_id カテゴリ化
        merged_df = self.data_merger.get_merged_df()
        
        le_horse = LabelEncoder().fit(merged_df['horse_id'])
        le_jockey = LabelEncoder().fit(merged_df['jockey_id'])
        le_trainer = LabelEncoder().fit(merged_df['trainer_id'])
        le_breeder = LabelEncoder().fit(merged_df['breeder_id'])
        self.le_horse = le_horse
        self.le_jockey = le_jockey
        self.le_trainer = le_trainer
        self.le_breeder = le_breeder
        
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
        
        
        mask_breeder = merged_df['breeder_id'].isin(le_breeder.classes_)
        new_breeder_id = merged_df['breeder_id'].mask(mask_breeder).dropna().unique()
        le_breeder.classes_ = np.concatenate([le_breeder.classes_, new_breeder_id])
        merged_df['breeder_id'] = le_breeder.transform(merged_df['breeder_id'])
        
        
        #horse_id, jockey_idをpandasのcategory型に変換
        merged_df['horse_id'] = merged_df['horse_id'].astype('category')
        merged_df['jockey_id'] = merged_df['jockey_id'].astype('category')
        merged_df['trainer_id'] = merged_df['trainer_id'].astype('category')
        merged_df['breeder_id'] = merged_df['breeder_id'].astype('category')
        
        self.processed_df = merged_df
        
        
    def dumminize(self):
        #そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        #列を一定にするため
        merged_df = self.data_merger.get_merged_df()
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
        
        self.processed_df = dumminized_df  