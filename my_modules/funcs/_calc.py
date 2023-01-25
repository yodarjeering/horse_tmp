


def calc_(all_results,kaime='tansho',odds_alpha=2,bet=100,is_all=True,round_list=['01'],verbose=True):
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
    else:
        print("No such kaime")
        return 


    for race_id in all_results.index:
        ar = all_results.loc[race_id]
        result_dict = calc_func(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=odds_alpha,bet=100,is_all=is_all,round_list=round_list)
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

def calc_tansho(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01']):

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




    
def calc_fukusho(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01']):
    
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

    
def calc_wide(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01']):


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


def calc_umaren(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01']):

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



def calc_umatan(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01']):

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

def calc_sanrenpuku(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01']):

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



def calc_sanrentan(ar,race_id,length,profit,tekichu,race_hit_dist,cant_buy_cnt,odds_alpha=2,bet=100,is_all=True,round_list=['01']):

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

