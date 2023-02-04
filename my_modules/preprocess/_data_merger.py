from sklearn.preprocessing import LabelEncoder
import pandas as pd



class DataMerger():
    
    
    def __init__(self, race_results_processor, horse_results_processor, peds_processor):
        self.race_results_processor = race_results_processor
        self.horse_results_processor = horse_results_processor
        self.peds_processor = peds_processor
        self.merged_df : pd.DataFrame() = None
        self.categorized_df : pd.DataFrame() = None
    
    
    def merge(self):
        race_results_processed = self.race_results_processor.get_processed_df()
        horse_results_processed = self.horse_results_processor.get_processed_df()
        peds_processed = self.peds_processor.get_processed_df()
        
        # race_results と　horse_results をマージ
        merged_df = race_results_processed.copy()
        """ 
            merged_df = merge(race_results_processed,horse_results_processed)
        """
        # for n_samples in n_samples_list:
        #     ## merge_all が　気に食わない, 将来的には Datamerger に実装
        #     merged_df = self.horse_results_processor.merge_all(merged_df, n_samples=n_samples)
        
        # 開催データをカテゴリ変数化する
        # merged_df.drop(['開催'], axis=1, inplace=True)
        
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
    

    
    