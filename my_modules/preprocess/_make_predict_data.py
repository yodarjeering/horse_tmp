import pandas as pd
from ._make_train_data import MakeTrainData
import numpy as np

class MakePredictData(MakeTrainData):
    
    
    def __init__(self,data_merger,COLUMNS):
        self.data_merger = data_merger
        self.processed_df = self.data_merger.get_merged_df()
        # 当日データでは, dumminaize のカラムが足りなくなってしまうので, 訓練データのカラムを追加
        self.COLUMNS = COLUMNS

    
    def get_processed_df(self):
        return self.processed_df
        

    # label encoder に再現性はない
    # したがって, 訓練データ作成時の label_encoderでなければカテゴリデータの学習結果が反映されない
    def set_lable_encoder(self,le_horse,le_jockey,le_trainer,le_owner,le_peds_dict):
        self.le_horse = le_horse
        self.le_jockey = le_jockey
        self.le_trainer = le_trainer
        self.le_owner = le_owner
        self.le_peds_dict = le_peds_dict


    #  この辺, いずれリファクタリングする
    #  継承の旨みが全く生かせてない
    def categorize_peds(self):
        merged_df = self.get_processed_df()
        peds_processor = self.data_merger.peds_processor
        peds_processed = peds_processor.processed_df
        le_peds_dict = self.le_peds_dict

        for column in peds_processed.columns:
            target_col = merged_df[column].astype(str).fillna('Na')
            # le_peds_dict[column] = LabelEncoder().fit(target_col)
            merged_df[column] = le_peds_dict[column].transform(target_col)
        
        print("--finish categorize peds--")
        self.le_peds_dict = le_peds_dict
        self.processed_df = merged_df
    

    # ここもリファクタリング必要
    def categorize_id(self):

        # horse_id, jockey_id, trainer_id, owner_id カテゴリ化
        merged_df = self.get_processed_df()

        # set_label_encoder 呼び出し後を仮定
        le_horse = self.le_horse
        le_jockey = self.le_jockey
        le_trainer = self.le_trainer
        le_owner = self.le_owner

        
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
        COLUMNS = self.COLUMNS
        merged_df = self.get_processed_df()
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

        # 当日のレース情報しかワンホットで作成されないため, 訓練データのカラム追加
        processed_df =  pd.DataFrame(merged_df,columns=COLUMNS)
        for col in dumminized_df.columns:
            processed_df[col] = dumminized_df[col]

        print("--finish dumminize--")
        processed_df.drop(['date'],axis=1,inplace=True)
        self.processed_df = processed_df.fillna(0)


            