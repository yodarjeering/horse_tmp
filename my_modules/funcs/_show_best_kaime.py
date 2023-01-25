from ._calc import calc_

def show_best_kaime(result_df,kaime,odds_alpha_list=[i for i in range(21)]):
    best_prf = -10**7
    best_prf_race = None

    best_recovery_rate = -10**7
    best_rec_race = None
    race_list = [str(i).zfill(2) for i in range(1,13)]

    for race_ in race_list:
        best_prf = -10**7
        best_prf_race = None
        best_accuracy = -10**7
        best_recovery_rate = -10**7

        for odds_alpha in odds_alpha_list:

            detail_dict  = calc_(result_df,kaime=kaime,odds_alpha=odds_alpha,is_all=False,round_list=[race_],verbose=False)
            prf = detail_dict['profit']
            recovery_rate = detail_dict['recovery_rate']
            accuracy_ = detail_dict['accuracy_'] 

            if best_prf<prf:
                best_prf = prf
                best_prf_odds = odds_alpha
                best_recovery_rate=recovery_rate
                best_accuracy = accuracy_
    
        print("---------------------")
        print(race_)
        print("odds_alpha         :",best_prf_odds)
        print("best_prf            :",best_prf)
        print("best_recovery_rate :",best_recovery_rate)
        print("best_accuracy      :",best_accuracy)
    