from numpy import True_
import pandas as pd
import time
import tqdm


class Simulator():
    
    def __init__(self, model, is_keiba_ai=False):
        
        self.model = model
        self.return_tables = None
        self.pred_df = None
        self.is_long = True
        self.is_keiba_ai = is_keiba_ai
    
    
    def return_pred_table(self,data_c,is_long=True):
        # is_long = True => test データ
        # is_long = False => 当日データ
        is_keiba_ai = self.is_keiba_ai
        if not is_long:
            if is_keiba_ai:
                scores = pd.Series(self.model.predict_proba(data_c.drop(['単勝'],axis=1))[:,1],index=data_c.index)
            else:
                scores = pd.Series(self.model.predict(data_c.drop(['単勝'], axis=1)),index=data_c.index)
        else:
            if is_keiba_ai:
                scores = pd.Series(self.model.predict_proba(data_c.drop(['date','rank','単勝'],axis=1))[:,1],index=data_c.index)
            else:
                scores = pd.Series(self.model.predict(data_c.drop(['date','rank','単勝'],axis=1)),index=data_c.index)
        pred = data_c[['馬番']].copy()
        pred['scores'] = scores
        pred = pred.sort_values('scores',ascending=False)
        return pred

#     odds以上の馬券しか買わない
    def get_result_df(self, data_c, return_tables, is_long=True, odds=2.0, bet = 100):
        race_id_list = sorted(list(set(data_c.index)))
        race_dict = {}

        for race_id in race_id_list:
            row_list = self.return_race_result(data_c,race_id,return_tables,is_long=is_long)
            if row_list==None:    
                continue
            race_dict[int(race_id)] = row_list
            
        all_result = pd.DataFrame(race_dict).T
        all_result.rename(columns={
            0:'pred_list',
            1:'actual_rank_list',
            2:'tansho_odds',
            3:'fukusho_odds',
            4:'umaren_odds',
            5:'wide_odds',
            6:'umatan_odds',
            7:'sanrenpuku_odds',
            8:'sanrentan_odds',
            9:'wide_comb',
            10:'odds_list'
            },inplace=True)
        return all_result
        

    def return_race_result(self, data_c ,race_id, return_tables,is_long=True):
        pred_df = self.return_pred_table(data_c.loc[race_id],is_long=is_long)
        pred_df = pred_df.loc[race_id]
        pred_df = pred_df.sort_values('scores',ascending=False)
        dc = data_c.loc[race_id]
        return_table  = return_tables.loc[race_id]
        
        
        pred_list = [int(pred_df['馬番'].iloc[i]) for i in range(len(pred_df))]

        score_1 = pred_df['scores'].iloc[0]
        score_2 = pred_df['scores'].iloc[1]
        is_same_score = False

        try:
        
            tansho_row = return_table[return_table[0]=='単勝']
            fukusho_row = return_table[return_table[0]=='複勝']
            umaren_row =  return_table[return_table[0]=='馬連']
            umatan_row =  return_table[return_table[0]=='馬単']
            wide_row =  return_table[return_table[0]=='ワイド']
            sanrentan_row =  return_table[return_table[0]=='三連単']
            sanrenpuku_row =  return_table[return_table[0]=='三連複']
            
            # odds 順番は予測した順
            odds_list = []
            for ub in pred_df['馬番'].tolist():
                odds_list.append(dc[dc['馬番']==ub]['単勝'].values[0])
        
            if score_1 == score_2:
                is_same_score =True
                
            # １着が同着    
            if int(tansho_row[1].str.count('br'))==1:
                actual_tmp0 = sanrentan_row[1].str.split('br').values[0][0]
                actual_tmp1 = sanrentan_row[1].str.split('br').values[0][1]
                actual_rank_list0 = list(map(int,actual_tmp0.split('→')))
                actual_rank_list1 = list(map(int,actual_tmp1.split('→')))
                actual_rank_list = [actual_rank_list0,actual_rank_list1]
                
                tansho_odds_list = tansho_row[2].str.split('br').values[0][0:3]
                tansho_odds_list = [i for i in tansho_odds_list if i!='']
                tansho_odds = list(map(lambda x: int(x.replace(',',''))/100 ,tansho_odds_list))
                
                umatan_odds_list = umatan_row[2].str.split('br').values[0][0:3]
                umatan_odds_list = [i for i in umatan_odds_list if i!='']
                umatan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,umatan_odds_list))
                
                sanrentan_odds_list = sanrentan_row[2].str.split('br').values[0][0:3]
                sanrentan_odds_list = [i for i in sanrentan_odds_list if i!='']
                sanrentan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrentan_odds_list))
        
                umaren_odds = int(umaren_row[2])/100
                sanrenpuku_odds = int(sanrenpuku_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:3]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:3]))
                
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:3]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                    
            # S2 2,3 番で同着
            elif int(umaren_row[1].str.count('br'))==1:
                actual_tmp0 = sanrentan_row[1].str.split('br').values[0][0]
                actual_tmp1 = sanrentan_row[1].str.split('br').values[0][1]
                actual_rank_list0 = list(map(int,actual_tmp0.split('→')))
                actual_rank_list1 = list(map(int,actual_tmp1.split('→')))
                actual_rank_list = [actual_rank_list0,actual_rank_list1]
                
                tansho_odds = int(tansho_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:3]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                
                umaren_odds_list = umaren_row[2].str.split('br').values[0][0:3]
                umaren_odds_list = [i for i in umaren_odds_list if i!='']
                umaren_odds = list(map(lambda x: int(x.replace(',',''))/100 ,umaren_odds_list))
                
                umatan_odds_list = umatan_row[2].str.split('br').values[0][0:3]
                umatan_odds_list = [i for i in umatan_odds_list if i!='']
                umatan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,umatan_odds_list))
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:3]))
                
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:3]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                    
                sanrenpuku_odds = int(sanrenpuku_row[2])/100
                sanrentan_odds_list = sanrentan_row[2].str.split('br').values[0][0:3]
                sanrentan_odds_list = [i for i in sanrentan_odds_list if i!='']
                sanrentan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrentan_odds_list))
            
            # S3 3着で同着
            elif int(sanrenpuku_row[1].str.count('br'))==1:
                actual_tmp0 = sanrentan_row[1].str.split('br').values[0][0]
                actual_tmp1 = sanrentan_row[1].str.split('br').values[0][1]
                actual_rank_list0 = list(map(int,actual_tmp0.split('→')))
                actual_rank_list1 = list(map(int,actual_tmp1.split('→')))
                actual_rank_list = [actual_rank_list0,actual_rank_list1]
                
                tansho_odds = int(tansho_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:4]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                umaren_odds = int(umaren_row[2])/100
                umatan_odds = int(umatan_row[2])/100
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:5]))
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:5]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                
                sanrenpuku_odds_list = sanrenpuku_row[2].str.split('br').values[0][0:3]
                sanrenpuku_odds_list = [i for i in sanrenpuku_odds_list if i!='']
                sanrenpuku_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrenpuku_odds_list))
                
                sanrentan_odds_list = sanrentan_row[2].str.split('br').values[0][0:3]
                sanrentan_odds_list = [i for i in sanrentan_odds_list if i!='']
                sanrentan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrentan_odds_list))
            
            # S4 通常の順位
            else: #int(fukusho_row[1].str.count('br'))==2:    
                actual_rank_list = list(map(int,sanrentan_row[1].str.split('→').values[0]))
                
                tansho_odds = int(tansho_row[2])/100
                umaren_odds = int(umaren_row[2])/100
                sanrenpuku_odds = int(sanrenpuku_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:3]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:3]))
                
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:3]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                
                umatan_odds = int(umatan_row[2])/100
                
                sanrentan_odds = int(sanrentan_row[2])/100
            
            # それ以外の例外は, 十分少ないものとし, シミュレーションとして考慮しない
        except Exception as e:
            print(e)
            print(race_id)
            return None
        
        
        return  [pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb,odds_list]

class TodaySimulator(Simulator):

    def __init__(self,model, is_keiba_ai=False):
        # super(TodaySimulator,self).__init__(model)
        super().__init__(model,is_keiba_ai)
        self.is_long = False

    def return_table_today(self,race_id_list):
        return_tables = {}
        for race_id in race_id_list:
            try:
                url = 'https://race.netkeiba.com/race/result.html?race_id='+race_id+'&amp;rf=race_submenu'
                dfs = pd.read_html(url)
                df = pd.concat([dfs[1], dfs[2]])
                df.index = [race_id] * len(df)
                return_tables[race_id] = df
                time.sleep(0.5)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break
            #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        # return_tables_df.index = return_tables_df.index.astype(int)
        self.return_tables = return_tables_df
        return return_tables_df

    def return_race_result(self,data_c,race_id,return_tables):
        dc = data_c.loc[race_id]
        pred_df = self.return_pred_table(data_c.loc[race_id],is_long=self.is_long)
        try:
            odds_list = []
            for ub in pred_df['馬番'].tolist():
                odds_list.append(dc[dc['馬番']==ub]['単勝'].values[0])
            
            return_table  = return_tables.loc[str(race_id)]
            pred_df = pred_df.loc[race_id]
            pred_df = pred_df.sort_values('scores',ascending=False)
            pred_list = [int(pred_df['馬番'].iloc[i]) for i in range(len(pred_df))]

            score_1 = pred_df['scores'].iloc[0]
            score_2 = pred_df['scores'].iloc[1]
            is_same_score = False

            
            tansho_row = return_table[return_table[0]=='単勝']
            fukusho_row = return_table[return_table[0]=='複勝']
            umaren_row =  return_table[return_table[0]=='馬連']
            umatan_row =  return_table[return_table[0]=='馬単']
            wide_row =  return_table[return_table[0]=='ワイド']
            sanrentan_row =  return_table[return_table[0]=='3連単']
            sanrenpuku_row =  return_table[return_table[0]=='3連複']
            
            actual_rank_list = list(map(int,sanrentan_row[1].str.split(' ').values[0]))
            
            odds_tmp =umaren_row[2].str.replace(',','')
            umaren_odds = int(odds_tmp.str.replace('円','').values[0])/100
            
            odds_tmp = tansho_row[2].str.replace(',','')
            tansho_odds = int(odds_tmp.str.replace('円','').values[0])/100
            
            odds_tmp = umatan_row[2].str.replace(',','')
            umatan_odds = int(odds_tmp.str.replace('円','').values[0])/100
            
            odds_tmp = sanrenpuku_row[2].str.replace(',','')
            sanrenpuku_odds = int(odds_tmp.str.replace('円','').values[0])/100
            
            odds_tmp = sanrentan_row[2].str.replace(',','')
            sanrentan_odds = int(odds_tmp.str.replace('円','').values[0])/100
            
            fukusho_odds_list = fukusho_row[2].str.split('円').values[0][0:3]
            fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
            fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
            wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('円').values[0][0:3]))
            
            tmp_list = list(map(int,wide_row[1].str.split(' ').tolist()[0]))
            wide_comb = []
            for i in range(0,len(tmp_list),2):
                wide_comb.append(tmp_list[i:i+2])
            
            if score_1 == score_2:
                is_same_score =True
            
        except Exception as e:
            print(e)
            print("race_id :",race_id)
            return None
        
        return  [pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb,odds_list]

    def get_result_df(self, data_c_list, return_tables, race_id_list, odds=1, bet=100):
        race_dict = {}

        for i, race_id in enumerate(race_id_list):
            row_list = self.return_race_result(data_c_list[i],race_id,return_tables)
            if row_list==None:
                print("race_id :",race_id)
                continue
            
            race_dict[int(race_id)] = row_list
        all_result = pd.DataFrame(race_dict).T
        all_result.rename(columns={
            0:'pred_list',
            1:'actual_rank_list',
            2:'tansho_odds',
            3:'fukusho_odds',
            4:'umaren_odds',
            5:'wide_odds',
            6:'umatan_odds',
            7:'sanrenpuku_odds',
            8:'sanrentan_odds',
            9:'wide_comb',
            10:'odds_list'
            },inplace=True)
        return all_result


