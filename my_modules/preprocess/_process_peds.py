import pandas as pd
from tqdm.notebook import tqdm as tqdm
import re
from sklearn.preprocessing import LabelEncoder

class Peds:

    def __init__(self, peds:pd.DataFrame):
        self.peds = peds
        self.peds_cat = pd.DataFrame() #after label encoding and transforming into category
        self.peds_re = pd.DataFrame()
        self.peds_vec = pd.DataFrame()
    
    
#     血統データが正規化されていないデータに対して, 正規化する関数
    def preprocessing(self):
        peds = self.peds.copy()
        error_idx_list = []
        for idx in tqdm(peds.index):
            for col in peds.columns:
            #     漢字 : 一-龥
                code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％一-龥\d]')
                try:
                    cleaned_text = code_regex.sub('', peds[col].loc[idx])
                    one_word = "".join(cleaned_text.split())
                    p_alphabet = re.compile('[a-zA-Z]+')
                    p_katakana = re.compile(r'[ァ-ヶー]+')
                    peds[col].loc[idx] = one_word
                    
                    if (not p_alphabet.fullmatch(one_word)) and not (p_katakana.fullmatch(one_word)):
                        peds[col].loc[idx] = re.sub('[a-zA-Z]+', '', one_word)
                except:
                    error_idx_list.append(idx)
        print("finish regularize")
        self.error_idx_list_r = error_idx_list
        self.peds_re = peds
        self.processed_df = peds

    
    def categorize(self):
        df = self.peds.copy()
        self.le_peds_dict = {}
        
        for column in df.columns:          
            self.le_peds_dict[column] = LabelEncoder()
            df[column] = self.le_peds_dict[column].fit_transform(df[column].fillna('Na'))

        print("finish categorize")
        self.peds_cat = df.astype('category')
        self.le_peds = self.le_peds_dict
        self.processed_df = df.peds_cat
        
        
#         血統データをベクトル化する関数
#         将来的には実装したい
    def vectorize(self):
        pass
    
