from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np



class DataMerger():
    
    
    def __init__(
        self, 
        race_results_processor, 
        horse_results_processor, 
        peds_processor
        ):
        self.race_results_processor = race_results_processor
        self.horse_results_processor = horse_results_processor
        self.peds_processor = peds_processor
        self.merged_df : pd.DataFrame() = None
        self.categorized_df : pd.DataFrame() = None
    
    
    def merge(self, n_samples_list=[5, 9, 'all']):
        race_results_processed = self.race_results_processor.processed_df
        horse_results_processed = self.horse_results_processor.processed_df
        peds_processed = self.peds_processor.processed_df
        
        # race_results と　horse_results をマージ
        merged_df = race_results_processed.copy()
        for n_samples in n_samples_list:
            ## merge_all が　気に食わない, 将来的には Datamerger に実装
            merged_df = self.horse_results_processor.merge_all(merged_df, n_samples=n_samples)
        merged_df.drop(['開催'], axis=1, inplace=True)
        
        # peds と merged_df をマージ
        merged_df.merge(peds_processed, left_on='horse_id', right_index=True,how='left',inplace=True)
        # 重複データを削除
        merged_df = merged_df[~merged_df.duplicated()]
        
        # 血統データない馬をマージした場合
        no_peds = merged_df[merged_df['peds_0'].isnull()]['horse_id'].unique()
        if len(self.no_peds) > 0:
            print('scrape peds at horse_id_list "no_peds"')
            print('no peds list',no_peds)
        
        self.merged_df = merged_df
        
    
    def get_merged_df(self):
        return self.merged_df
    
    
    def get_categorized_df(self):
        return self.categorized_df

    
    def process_categorical(self):
        # 血統データカテゴリ変数化
        peds_processor = self.peds_processor
        peds_processed = peds_processor.processed_df
        
        le_peds_dict = {}
        for column in peds_processed.columns:
            le_peds_dict[column] = LabelEncoder()
            peds_processed[column] = le_peds_dict[column].fit_transform(peds_processed[column].fillna('Na'))
        
        print("finish categorize peds")
        self.le_peds_dict = le_peds_dict
        
        # horse_id, jockey_id, trainer_id, breeder_id カテゴリ化
        merged_df = self.get_merged_df()
        
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
        
        
        #そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        #列を一定にするため
        weathers = merged_df['weather'].unique()
        race_types = merged_df['race_type'].unique()
        ground_states = merged_df['ground_state'].unique()
        sexes = merged_df['性'].unique()
        merged_df['weather'] = pd.Categorical(merged_df['weather'], weathers)
        merged_df['race_type'] = pd.Categorical(merged_df['race_type'], race_types)
        merged_df['ground_state'] = pd.Categorical(merged_df['ground_state'], ground_states)
        merged_df['性'] = pd.Categorical(merged_df['性'], sexes)
        merged_df = pd.get_dummies(merged_df, columns=['weather', 'race_type', 'ground_state', '性'])
        self.categorized_df = merged_df    