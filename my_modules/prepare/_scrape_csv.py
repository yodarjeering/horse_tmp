from tqdm.auto import tqdm
import pandas as pd
import time
import urllib.request
import requests
from bs4 import BeautifulSoup
import re
SLEEP_TIME = 5 # サーバに負担をかけないよう, SLEEP _TIME だけ待つ

def scrape_horse_results(horse_id_list:list):
        #horse_idをkeyにしてDataFrame型を格納
        # horse_id : int
        horse_results = {}
        for horse_id in tqdm(horse_id_list):
            try:
                url = 'https://db.netkeiba.com/horse/' + str(horse_id)
                df = pd.read_html(url)[3]
                #受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
                if df.columns[0]=='受賞歴':
                    df = pd.read_html(url)[4]
                df.index = [horse_id] * len(df)
                horse_results[horse_id] = df
                time.sleep(SLEEP_TIME)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
        #pd.DataFrame型にして一つのデータにまとめる        
        horse_results_df = pd.concat([horse_results[key] for key in horse_results])
        return horse_results_df

def scrape_peds(horse_id_list:list):
        # horse_id : int
        
        peds_dict = {}
        for horse_id in tqdm(horse_id_list):
#         for horse_id in horse_id_list:
            try:
                url = "https://db.netkeiba.com/horse/ped/" + str(horse_id)
            
                df = pd.read_html(url)[0]

                #重複を削除して1列のSeries型データに直す
                generations = {}
                for i in reversed(range(5)):
                    generations[i] = df[i]
                    df.drop([i], axis=1, inplace=True)
                    df = df.drop_duplicates()
                ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)
                peds_dict[horse_id] = ped.reset_index(drop=True)
                
                time.sleep(SLEEP_TIME)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break

        #列名をpeds_0, ..., peds_61にする
        peds_df = pd.concat([peds_dict[key] for key in peds_dict],axis=1).T.add_prefix('peds_')
        peds_df.index =peds_df.index.astype(int)
        return peds_df

def scrape_return_tables(race_id_list:list):
        # race_id : int
        """
        払い戻し表データをスクレイピングする関数

        Parameters:
        ----------
        race_id_list : list
            レースIDのリスト

        Returns:
        ----------
        return_tables_df : pandas.DataFrame
            全払い戻し表データをまとめてDataFrame型にしたもの
        """

        return_tables = {}
        for race_id in tqdm(race_id_list):
            try:
                url = "https://db.netkeiba.com/race/" + str(race_id)

                #普通にスクレイピングすると複勝やワイドなどが区切られないで繋がってしまう。
                #そのため、改行コードを文字列brに変換して後でsplitする
                f = urllib.request.urlopen(url)
                html = f.read()
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(html)

                #dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
                df = pd.concat([dfs[1], dfs[2]])
                df.index = [race_id] * len(df)
                return_tables[race_id] = df
                time.sleep(SLEEP_TIME)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break


        #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        return return_tables_df

def scrape_race_results(race_id_list:list):
        #race_idをkeyにしてDataFrame型を格納
        # race_id : int
        race_results = {}
        for race_id in tqdm(race_id_list):
            time.sleep(SLEEP_TIME)
            try:
                url = "https://db.netkeiba.com/race/" + str(race_id)
                #メインとなるテーブルデータを取得
                df = pd.read_html(url)[0]
                html = requests.get(url)
                html.encoding = "EUC-JP"
                soup = BeautifulSoup(html.text, "html.parser")

                #天候、レースの種類、コースの長さ、馬場の状態、日付をスクレイピング
                texts = (
                    soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
                    + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
                )
                info = re.findall(r'\w+', texts)
                for text in info:
                    if text in ["芝", "ダート"]:
                        df["race_type"] = [text] * len(df)
                    if "障" in text:
                        df["race_type"] = ["障害"] * len(df)
                    if "m" in text:
                        df["course_len"] = [int(re.findall(r"\d+", text)[0])] * len(df)
                    if text in ["良", "稍重", "重", "不良"]:
                        df["ground_state"] = [text] * len(df)
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df["weather"] = [text] * len(df)
                    if "年" in text:
                        df["date"] = [text] * len(df)

                #馬ID、騎手ID, owner_id, trainer_idをスクレイピング
                horse_id_list = []
                horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a",
                    attrs={"href": re.compile("^/horse")}
                )
                for a in horse_a_list:
                    horse_id = re.findall(r"\d+", a["href"])
                    horse_id_list.append(horse_id[0])
                jockey_id_list = []
                jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/jockey")}
                )
                for a in jockey_a_list:
                    jockey_id = re.findall(r"\d+", a["href"])
                    jockey_id_list.append(jockey_id[0])

                ############
                trainer_id_list = []
                trainer_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/trainer")}
                )
                for a in trainer_a_list:
                    trainer_id = re.findall(r"\d+", a["href"])
                    trainer_id_list.append(trainer_id[0])
                
                ###########
                owner_id_list = []
                owner_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/owner")}
                )
                for a in owner_a_list:
                    owner_id = re.findall(r"\d+", a["href"])
                    owner_id_list.append(owner_id[0])
                
                df["horse_id"] = horse_id_list
                df["jockey_id"] = jockey_id_list
                df["trainer_id"] = trainer_id_list
                df["owner_id"] = owner_id_list
                
                
                #インデックスをrace_idにする
                df.index = [race_id] * len(df)
                race_results[race_id] = df

            #存在しないrace_idを飛ばす
            except IndexError:
                print("IndexError")
                continue
            #wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(e)
                break


        #pd.DataFrame型にして一つのデータにまとめる
        race_results_df = pd.concat([race_results[key] for key in race_results])
        return race_results_df