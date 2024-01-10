from itertools import combinations

# 将来的にはrace type 別の的中率分布作りたい
RACE_TYPE_LIST = [
    'new_horse',
    'not_win',
    'won1',
    'won2',
    'won3',
    'open',
    'g1',
    'g2',
    'g3',
    'obstacle'
]


def calc_(all_results,kaime='tansho',odds_alpha=2,bet=100,is_all=True,round_list=['01'],verbose=True,num_buy=3,race_type_list=['open']):

    variable_dict = {
        'length':0,         # race の長さ
        'tekichu':0,        # 的中した数
        'profit':0,         # 利益 (betした経費は計算済み)
        'cant_buy_cnt':0,   # odds_filter 以下のオッズで, 買えなかったレースの数
        'bought_num':1      # 買い点数, box買い以外は, 1, 
    }

    # race_round 別の的中数分布
    race_hit_dist = {'{}'.format(str(i).zfill(2)):0 for i in range(1,13)}  
    # race_class 別の的中数分布
    race_hit_type_dist = {'{}'.format(rt):0 for rt in RACE_TYPE_LIST}
    # box 買いの時は的中率, 回収率などの計算が他とは異なる
    is_box = False

    if kaime=='tansho':
        calc_func = calc_tansho
    elif kaime=='fukusho':
        calc_func = calc_fukusho
    elif kaime=='wide':
        calc_func = calc_wide_box
        num_buy = 2
    elif kaime=='umatan':
        calc_func = calc_umatan
    elif kaime=='umaren':
        calc_func = calc_umaren_box
        num_buy = 2
    elif kaime=='sanrentan':
        calc_func = calc_sanrentan
    elif kaime=='sanrenpuku':
        calc_func = calc_sanrenpuku_box
        num_buy = 3
    elif kaime=='wide_box':
        calc_func = calc_wide_box
        is_box = True
    elif kaime=='umaren_box':
        calc_func = calc_umaren_box
        is_box = True
    elif kaime=='sanrenpuku_box':
        calc_func = calc_sanrenpuku_box
        is_box = True
    elif kaime=='tansho_proper':
        calc_func = calc_tansho_proper
    else:
        print("No such kaime")
        return 


    for race_id in all_results.index:
        ar = all_results.loc[race_id]
        result_dict = calc_func(
            ar,                     
            race_id,        
            variable_dict,          # 計算する変数を辞書型で管理
            race_hit_dist,          # race_round ごとの的中率分布の計算用
            race_hit_type_dist,     # race_type 別の的中率分布計算用 *未実装
            odds_alpha=odds_alpha,  # 購入する単勝オッズの閾値, この単勝オッズより下なら買わない, どの買目でも単勝オッズで判断する点に注意
            bet=bet,                # 掛金
            is_all=is_all,          
            round_list=round_list,
            num_buy=num_buy         # box 買いの時, 何点購入するか
            )
        variable_dict['length'] = result_dict['length']
        variable_dict['profit'] = result_dict['profit']
        variable_dict['tekichu'] = result_dict['tekichu']
        variable_dict['cant_buy_cnt'] = result_dict['cant_buy_cnt']

        if is_box:
            variable_dict['bought_num'] = result_dict['bought_num']

        race_hit_dist = result_dict['race_hit_dist']
        race_hit_type_dist = result_dict['race_hit_type_dist']


    length = variable_dict['length']                # raceの長さ
    profit = variable_dict['profit']                # 利益
    tekichu = variable_dict['tekichu']              # 的中レース数
    cant_buy_cnt = variable_dict['cant_buy_cnt']    # 指定したオッズ以下で買わなかったレースの数
    race_num = length                               
    recovery_rate = (profit/(bet*race_num)+1)*100   # 回収率 (%)
    accuracy_ = (tekichu/race_num)*100              # 的中率
    bought_num = variable_dict['bought_num']        # 買った点数, box買い以外は 1

    if is_box:
        filtered_race_num = bought_num-1

    else:
        filtered_race_num = race_num-cant_buy_cnt
    
    if filtered_race_num!=0:
        filtered_accuracy_ = tekichu/filtered_race_num
        if is_box:
            mean_profit = profit/bought_num
        else:
            mean_profit = profit/filtered_race_num
    # 賭ける対象のレースがすべてodds_alphaより小さかった場合
    else:
        filtered_accuracy_ = -1
        mean_profit = -1
    
    if verbose:
        
        print("収益               : {:>15.2f} 円".format(profit))
        print("平均収益           : {:>15.2f} 円".format(mean_profit))
        print("回収率             : {:>15.2f} %".format(recovery_rate))
        print('的中率             : {:>15.2f} %'.format(accuracy_))
        print("odds filter 的中率 : {:>15.2f} %".format(filtered_accuracy_*100))
        if is_box:
            print("賭けた点数の回収率 : {:>15.2f} %".format((profit/(bet*bought_num)+1)*100))
            print("賭けた点数の的中率 : {:>15.2f} %".format((tekichu/bought_num)*100))
        print("レース数           : {:>15}".format(race_num))
        print("賭けた点数         : {:>15}".format(bought_num))
        print('race dist',race_hit_dist)
        print("race type dist",race_hit_type_dist)

    # １レースあたりの平均収益とかあるといいかも
    detail_dict = {
        'accuracy_':accuracy_,
        'filtered_accuracy_':filtered_accuracy_,
        'filtered_race_num':filtered_race_num,
        'race_num':race_num,
        'profit':profit,
        'mean_profit':mean_profit,
        'recovery_rate':recovery_rate,
        'race_hit_dist':race_hit_dist,
        'race_hit_type_dist':race_hit_type_dist,
        'bought_num':bought_num
    }

    return detail_dict

def calc_tansho(ar,race_id,variable_dict,race_hit_dist,race_hit_type_dist,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):
    
    length = variable_dict['length']
    profit = variable_dict['profit'] 
    tekichu = variable_dict['tekichu'] 
    cant_buy_cnt = variable_dict['cant_buy_cnt'] 
    # race_class = ar['race_class']

    if is_all or (str(race_id)[-2:] in round_list):# or (race_class in round_list):
        length += 1
        is_buy = True
        pred_list = ar['pred_list']
        actual_list = ar['actual_rank_list']
        tansho_odds = ar['tansho_odds']
        pred_odds = ar['odds_list'][0]
            
        if pred_odds>=odds_alpha:
            profit -= bet
        else:
            is_buy=False
            cant_buy_cnt += 1
        
            
        if pred_list[0]==actual_list[0] and is_buy:
            tekichu+=1
            profit += bet*tansho_odds
            race_hit_dist[str(race_id)[-2:]] += 1
            # race_hit_type_dist[race_class] += 1

    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'race_hit_type_dist':race_hit_type_dist
        }


def calc_fukusho(ar,race_id,variable_dict,race_hit_dist,race_hit_type_dist,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):
    length = variable_dict['length']
    profit = variable_dict['profit'] 
    tekichu = variable_dict['tekichu'] 
    cant_buy_cnt = variable_dict['cant_buy_cnt'] 
    # race_class = ar['race_class']

    if is_all or (str(race_id)[-2:] in round_list):# or (race_class in round_list):
        length += 1
        is_buy = True
        pred_list = ar['pred_list']
        actual_list = ar['actual_rank_list']
        fukusho_odds = ar['fukusho_odds']
        fukusho_ken = 3
        pred_odds = ar['odds_list'][0]
        
        if pred_odds>=odds_alpha:
            profit -= bet
        else:
            is_buy=False
            cant_buy_cnt += 1
        

        if len(fukusho_odds)==2:
            fukusho_ken = 2

        if pred_list[0] in actual_list[0:fukusho_ken] and is_buy:

            tekichu_index = actual_list[0:fukusho_ken].index(pred_list[0])
            tekichu+=1
            profit += bet*fukusho_odds[tekichu_index]
            race_hit_dist[str(race_id)[-2:]] += 1
            # race_hit_type_dist[race_class] += 1

    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'race_hit_type_dist':race_hit_type_dist
        }


def calc_umatan(ar,race_id,variable_dict,race_hit_dist,race_hit_type_dist,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):
    length = variable_dict['length']
    profit = variable_dict['profit'] 
    tekichu = variable_dict['tekichu'] 
    cant_buy_cnt = variable_dict['cant_buy_cnt'] 
    # race_class = ar['race_class']

    if is_all or (str(race_id)[-2:] in round_list):# or (race_class in round_list):
        length+=1
        is_buy = True
        pred_list = ar['pred_list'][0:2]
        actual_list = ar['actual_rank_list'][0:2]
        umatan_odds = ar['umatan_odds']
        pred_odds = ar['odds_list'][0]
        
        if pred_odds>=odds_alpha:
            profit -= bet
        else:
            is_buy=False
            cant_buy_cnt += 1

            
        if  pred_list == actual_list and is_buy:
            tekichu+=1
            profit += bet*umatan_odds
            race_hit_dist[str(race_id)[-2:]] += 1
            # race_hit_type_dist[race_class] += 1

    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'race_hit_type_dist':race_hit_type_dist
        }



def calc_sanrentan(ar,race_id,variable_dict,race_hit_dist,race_hit_type_dist,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):
    length = variable_dict['length']
    profit = variable_dict['profit'] 
    tekichu = variable_dict['tekichu'] 
    cant_buy_cnt = variable_dict['cant_buy_cnt'] 
    # race_class = ar['race_class']

    if is_all or (str(race_id)[-2:] in round_list):#or (race_class in round_list):
        length+=1
        is_buy = True
        pred_list = ar['pred_list'][0:3]
        actual_list = ar['actual_rank_list'][0:3]
        sanrentan_odds = ar['sanrentan_odds']
        pred_odds = ar['odds_list'][0]
        
        if pred_odds>=odds_alpha:
            profit -= bet
        else:
            is_buy=False
            cant_buy_cnt += 1
            
        if pred_list == actual_list and is_buy:
            tekichu+=1
            profit += bet*sanrentan_odds
            race_hit_dist[str(race_id)[-2:]] += 1
            # race_hit_type_dist[race_class] += 1

    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'race_hit_type_dist':race_hit_type_dist
        }


##  num_buy 引数 あとで全ての関数に適用
## こちらの関数の方が汎用性高いので, 将来的に calc_~box を calc_~と置き換える

def calc_sanrenpuku_box(ar,race_id,variable_dict,race_hit_dist,race_hit_type_dist,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=6):
    length = variable_dict['length']
    profit = variable_dict['profit'] 
    tekichu = variable_dict['tekichu'] 
    cant_buy_cnt = variable_dict['cant_buy_cnt'] 
    bought_num = variable_dict['bought_num']
    # race_class = ar['race_class']
    

    if is_all or (str(race_id)[-2:] in round_list):# or (race_class in round_list):
        length+=1
        is_buy = True
        actual_list = sorted(ar['actual_rank_list'][0:3])
        sanrenpuku_odds = ar['sanrenpuku_odds']
        pred_odds = ar['odds_list'][0]
        ##########################
        # len(box_list) = num_buyC3 通り
        # num_buy : 予測上位 num_buy 番目まで買うか
        pred_list = sorted(ar['pred_list'][0:num_buy])
        box_list = list(map(list,combinations(pred_list,3)))
        low_odds_num = 0
        
        

        # {'馬番' : odds} となる dict 作成
        pred_odds_dict = {}
        for i,umaban in enumerate(pred_list):
            pred_odds_dict[umaban] = ar['odds_list'][i]

        for comb in box_list:

            for umaban in comb:
                
                # 三連複購入の買い目の中に, オッズ低い馬がいたら, 買わない
                if pred_odds_dict[umaban]<odds_alpha:
                    is_buy=False
                    cant_buy_cnt += 1
                    low_odds_num += 1
                    break # <= 途中でブレークっておかしくね？

            if is_buy:
                profit -= bet

            if comb == actual_list and is_buy:
                tekichu+=1
                profit += bet*sanrenpuku_odds
                race_hit_dist[str(race_id)[-2:]] += 1
                # race_hit_type_dist[race_class] += 1
            
        # odds_alpha 以上を含む最終的な, 購入点数
        bought_num += (len(box_list) - low_odds_num)
    
    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'race_hit_type_dist':race_hit_type_dist,
        'bought_num':bought_num
        }

def calc_umaren_box(ar,race_id,variable_dict,race_hit_dist,race_hit_type_dist,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):
    
    length = variable_dict['length']
    profit = variable_dict['profit'] 
    tekichu = variable_dict['tekichu'] 
    cant_buy_cnt = variable_dict['cant_buy_cnt'] 
    bought_num = variable_dict['bought_num']
    # race_class = ar['race_class']

    if is_all or (str(race_id)[-2:] in round_list):# or (race_class in round_list):
        length+=1
        is_buy = True
        actual_list = sorted(ar['actual_rank_list'][0:2])
        umaren_odds = ar['umaren_odds']
        pred_odds = ar['odds_list'][0]
        ##########################
        # len(box_list) = num_buyC2 通り
        # num_buy : 予測上位 num_buy 番目まで買うか
        pred_list = sorted(ar['pred_list'][0:num_buy])
        box_list = list(map(list,combinations(pred_list,2)))
        low_odds_num = 0

        # {'馬番' : odds} となる dict 作成
        pred_odds_dict = {}
        for i,umaban in enumerate(pred_list):
            pred_odds_dict[umaban] = ar['odds_list'][i]

        for comb in box_list:

            for umaban in comb:
                
                # 馬連購入の買い目の中に, オッズ低い馬がいたら, 買わない
                if pred_odds_dict[umaban]<odds_alpha:
                    is_buy=False
                    cant_buy_cnt += 1
                    low_odds_num += 1
                    break

            if is_buy:
                profit -= bet

            if comb == actual_list and is_buy:
                tekichu+=1
                profit += bet*umaren_odds
                race_hit_dist[str(race_id)[-2:]] += 1
                # race_hit_type_dist[race_class] += 1
        
    # odds_alpha 以上を含む最終的な, 購入点数
        bought_num += (len(box_list) - low_odds_num)
        
    # calc_ 関数で計算される的中率は, 全レース数の的中率
    # 別途, 賭けたレース数のうちの的中率も計算する必要あり -> tekichu/bought_num で計算できる
    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'race_hit_type_dist':race_hit_type_dist,
        'bought_num':bought_num
        }

def calc_wide_box(ar,race_id,variable_dict,race_hit_dist,race_hit_type_dist,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):

    length = variable_dict['length']
    profit = variable_dict['profit'] 
    tekichu = variable_dict['tekichu'] 
    cant_buy_cnt = variable_dict['cant_buy_cnt'] 
    bought_num = variable_dict['bought_num']
    # race_class = ar['race_class']

    if is_all or (str(race_id)[-2:] in round_list):# or (race_class in round_list):
        length+=1
        is_buy = True
        wide_comb = ar['wide_comb']
        wide_odds = ar['wide_odds']
        pred_odds = ar['odds_list'][0]
        ##########################
        # len(box_list) = num_buyC2 通り
        # num_buy : 予測上位 num_buy 番目まで買うか
        pred_list = sorted(ar['pred_list'][0:num_buy])
        # box_list : ex)  [[2, 5, 11], [2, 5, 12], [2, 11, 12], [5, 11, 12]]
        box_list = list(map(list,combinations(pred_list,2)))
        low_odds_num = 0

        # {'馬番' : odds} となる dict 作成
        pred_odds_dict = {}
        for i,umaban in enumerate(pred_list):
            pred_odds_dict[umaban] = ar['odds_list'][i]
        
        for comb in box_list:
            for umaban in comb:
                
                # 馬連購入の買い目の中に, オッズ低い馬がいたら, 買わない
                if pred_odds_dict[umaban]<odds_alpha:
                    is_buy=False
                    cant_buy_cnt += 1
                    low_odds_num += 1
                    break

            if is_buy:
                profit -= bet

            if comb in wide_comb and is_buy:
                tekichu+=1
                tekichu_index = wide_comb.index(comb)
                profit += bet*wide_odds[tekichu_index]
                race_hit_dist[str(race_id)[-2:]] += 1
                # race_hit_type_dist[race_class] += 1
        
    # odds_alpha 以上を含む最終的な, 購入点数
        bought_num += (len(box_list) - low_odds_num)
        
    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'race_hit_type_dist':race_hit_type_dist,
        'bought_num':bought_num
        }

# odds が低い馬には掛金大きく, odds が高い馬は掛金小さく
# bet = X / odds
def calc_tansho_proper(ar,race_id,variable_dict,race_hit_dist,race_hit_type_dist,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):
    
    # 1万円 獲得できるように, 掛金を調整する
    Expected_price = 10000
    length = variable_dict['length']
    profit = variable_dict['profit'] 
    tekichu = variable_dict['tekichu'] 
    cant_buy_cnt = variable_dict['cant_buy_cnt'] 
    # race_class = ar['race_class']

    if is_all or (str(race_id)[-2:] in round_list):# or (race_class in round_list):
        length += 1
        is_buy = True
        pred_list = ar['pred_list']
        actual_list = ar['actual_rank_list']
        tansho_odds = ar['tansho_odds']
        pred_odds = ar['odds_list'][0]

        # 払い戻し金額が, 1万円で一定となるように掛金を調整
        bet = Expected_price/pred_odds
        
        if bet<100:
            bet = 100

        profit -= bet
            
        if pred_list[0]==actual_list[0] and is_buy:
            tekichu+=1
            profit += bet*tansho_odds
            race_hit_dist[str(race_id)[-2:]] += 1
            # race_hit_type_dist[race_class] += 1

    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'race_hit_type_dist':race_hit_type_dist
        }

def calc_sanrentan_nagashi():
    pass