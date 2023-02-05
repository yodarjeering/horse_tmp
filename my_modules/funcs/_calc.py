from itertools import combinations


def calc_(all_results,kaime='tansho',odds_alpha=2,bet=100,is_all=True,round_list=['01'],verbose=True,num_buy=3):
    length = 0
    tekichu = 0
    profit = 0
    bet = 100
    cant_buy_cnt = 0
    race_hit_dist = {'{}'.format(str(i).zfill(2)):0 for i in range(1,13)}

    if kaime=='tansho':
        calc_func = calc_tansho
    elif kaime=='fukusho':
        calc_func = calc_fukusho
    elif kaime=='wide':
        calc_func = calc_wide
    elif kaime=='umatan':
        calc_func = calc_umatan
    elif kaime=='umaren':
        calc_func = calc_umaren
    elif kaime=='sanrentan':
        calc_func = calc_sanrentan
    elif kaime=='sanrenpuku':
        calc_func = calc_sanrenpuku
    elif kaime=='wide_box':
        calc_func = calc_wide_box
    elif kaime=='umaren_box':
        calc_func = calc_umaren_box
    elif kaime=='sanrenpuku_box':
        calc_func = calc_sanrenpuku_box
    else:
        print("No such kaime")
        return 


    for race_id in all_results.index:
        ar = all_results.loc[race_id]
        result_dict = calc_func(
            ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,
            odds_alpha=odds_alpha,
            bet=bet,
            is_all=is_all,
            round_list=round_list,
            num_buy=3
            )
        length = result_dict['length']
        profit = result_dict['profit']
        tekichu = result_dict['tekichu']
        cant_buy_cnt = result_dict['cant_buy_cnt']
        race_hit_dist = result_dict['race_hit_dist']

    # -----------------------------------
    # 回収率 = (収益)/(掛金)
    #        = profit/(bet*length)
    # 経費はすでに計算されているので, 収益率の方が正しい
    # -----------------------------------

    recovery_rate = (profit/(bet*length))*100
    if verbose:
        print('的中率             {0}'.format(tekichu/length))
        print("odds filter 的中率 {0}".format(tekichu/(length-cant_buy_cnt)))
        print("収益               {0} 円".format(profit))
        print("回収率             {0}".format(recovery_rate))
        print("レース数           {0}".format(length))
        print("賭けたレース数     {0}".format(length-cant_buy_cnt))
        print('race dist',race_hit_dist)

    accuracy_ = tekichu/length
    detail_dict = {
        'accuracy_':accuracy_,
        'profit':profit,
        'recovery_rate':recovery_rate,
        'race_hit_dist':race_hit_dist
    }

    return detail_dict

def calc_tansho(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):

    if is_all or str(race_id)[-2:] in round_list:
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

    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        }




    
def calc_fukusho(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):
    
    if is_all or str(race_id)[-2:] in round_list:
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

    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        }

    
def calc_wide(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):


    if is_all or str(race_id)[-2:] in round_list:
        length+=1
        is_buy = True
        pred_list = ar['pred_list']
        wide_comb = ar['wide_comb']
        wide_odds = ar['wide_odds']
        pred_odds = ar['odds_list'][0]
        
        if pred_odds>=odds_alpha:
            profit -= bet
        else:
            is_buy=False
            cant_buy_cnt += 1
            
        if sorted(pred_list[0:2]) in wide_comb and is_buy:
            tekichu_index = wide_comb.index(sorted(pred_list[0:2]))
            tekichu+=1
            profit += bet*wide_odds[tekichu_index]
            race_hit_dist[str(race_id)[-2:]] += 1

    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        }


def calc_umaren(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):

    if is_all or str(race_id)[-2:] in round_list:
        length+=1
        is_buy = True
        pred_list = sorted(ar['pred_list'][0:2])
        actual_list = sorted(ar['actual_rank_list'][0:2])
        umaren_odds = ar['umaren_odds']
        pred_odds = ar['odds_list'][0]
        
        if pred_odds>=odds_alpha:
            profit -= bet
        else:
            is_buy=False
            cant_buy_cnt += 1

        if  pred_list == actual_list and is_buy:
            tekichu+=1
            profit += bet*umaren_odds
            race_hit_dist[str(race_id)[-2:]] += 1
        
    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        }



def calc_umatan(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):

    if is_all or str(race_id)[-2:] in round_list:
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
    
    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        }

def calc_sanrenpuku(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):

    if is_all or str(race_id)[-2:] in round_list:
        length+=1
        is_buy = True
        pred_list = sorted(ar['pred_list'][0:3])
        actual_list = sorted(ar['actual_rank_list'][0:3])
        sanrenpuku_odds = ar['sanrenpuku_odds']
        pred_odds = ar['odds_list'][0]
        
        if pred_odds>=odds_alpha:
            profit -= bet
        else:
            is_buy=False
            cant_buy_cnt += 1

        if pred_list == actual_list and is_buy:
            tekichu+=1
            profit += bet*sanrenpuku_odds
            race_hit_dist[str(race_id)[-2:]] += 1
    
    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        }



def calc_sanrentan(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):

    if is_all or str(race_id)[-2:] in round_list:
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

    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        }

##  num_buy 引数 あとで全ての関数に適用
## こちらの関数の方が汎用性高いので, 将来的に calc_~box を calc_~と置き換える
def calc_sanrenpuku_box(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=6):

    bought_num = 0
    if is_all or str(race_id)[-2:] in round_list:
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
                    break

            if is_buy:
                profit -= bet

            if comb == actual_list and is_buy:
                tekichu+=1
                profit += bet*sanrenpuku_odds
                race_hit_dist[str(race_id)[-2:]] += 1
            
        # odds_alpha 以上を含む最終的な, 購入点数
        bought_num = len(box_list) - low_odds_num
    
    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'bought_num':bought_num
        }

def calc_umaren_box(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):

    bought_num = 0

    if is_all or str(race_id)[-2:] in round_list:
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
        
    # odds_alpha 以上を含む最終的な, 購入点数
        bought_num = len(box_list) - low_odds_num
        
    # calc_ 関数で計算される的中率は, 全レース数の的中率
    # 別途, 賭けたレース数のうちの的中率も計算する必要あり -> tekichu/bought_num で計算できる
    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'bought_num':bought_num
        }

def calc_wide_box(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01'],num_buy=3):

    bought_num = 0
    if is_all or str(race_id)[-2:] in round_list:
        length+=1
        is_buy = True
        wide_comb = ar['wide_comb']
        wide_odds = ar['wide_odds']
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

            if comb in wide_comb and is_buy:
                tekichu+=1
                tekichu_index = wide_comb.index(comb)
                profit += bet*wide_odds[tekichu_index]
                race_hit_dist[str(race_id)[-2:]] += 1
        
    # odds_alpha 以上を含む最終的な, 購入点数
        bought_num = len(box_list) - low_odds_num
        
    return {
        'length':length,
        'profit':profit,
        'tekichu':tekichu,
        'cant_buy_cnt':cant_buy_cnt,
        'race_hit_dist':race_hit_dist,
        'bought_num':bought_num
        }

def calc_tansho_proper():
    pass

def calc_sanrentan_nagashi():
    pass