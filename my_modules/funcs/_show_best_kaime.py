from hyperopt.pyll.base import Raise
from ._calc import calc_

RACE_TYPE_LIST = [
    'new_horse',
    'not_win',
    'won1',
    'won2',
    'won3',
    'open',
#  'g1',
#  'g2',
#  'g3',
    'obstacle'
]

def show_best_kaime(result_df,kaime,odds_alpha_list=[i for i in range(21)],num_buy=1,is_round=True):
    best_prf = -10**7
    best_prf_race = None
    best_recovery_rate = -10**7
    best_rec_race = None
    race_list = [str(i).zfill(2) for i in range(1,13)]

    if is_round:
        round_list = race_list
    else:
        round_list = RACE_TYPE_LIST

    for race_ in round_list:
        best_prf = -10**7
        best_prf_race = None
        best_accuracy = -10**7
        best_filtered_accuracy = -10**7
        best_race_num = -10**7
        best_filtered_race_num = -10**7
        best_recovery_rate = -10**7
        best_mean_profit = -10**7
        bought_num = 1

        for odds_alpha in odds_alpha_list:

            detail_dict  = calc_(result_df,kaime=kaime,odds_alpha=odds_alpha,is_all=False,round_list=[race_],verbose=False)
            prf = detail_dict['profit']
            mean_profit = detail_dict['mean_profit']
            recovery_rate = detail_dict['recovery_rate']
            accuracy_ = detail_dict['accuracy_'] 
            filtered_accuracy_ = detail_dict['filtered_accuracy_']
            race_num = detail_dict['race_num']
            filtered_race_num = detail_dict['filtered_race_num']

            if best_prf<prf:
                best_prf = prf
                best_mean_profit = mean_profit
                best_prf_odds = odds_alpha
                best_recovery_rate=recovery_rate
                best_accuracy = accuracy_
                best_filtered_accuracy = filtered_accuracy_
                best_race_num = race_num
                best_filtered_race_num = filtered_race_num
                bought_num = detail_dict['bought_num']

        print("---------------------")
        print(race_)
        print("odds_alpha             :{:>15}".format(best_prf_odds))
        print("best_prf               :{:>15.2f} 円".format(best_prf))
        print("best_mean_prf          :{:>15.2f} 円".format(best_mean_profit))
        print("best_recovery_rate     :{:>15.2f} %".format(best_recovery_rate))
        print("best_accuracy          :{:>15.2f} %".format(best_accuracy*100))
        print("best_filtered_accuracy :{:>15.2f} %".format(best_filtered_accuracy*100))
        print("total race_num         :{:>15}".format(best_race_num))
        print("filterd_race_num       :{:>15}".format(best_filtered_race_num))
        print("bought_num             :{:>15}".format(bought_num))


    
    