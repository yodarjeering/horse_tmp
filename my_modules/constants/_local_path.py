import os
import dataclasses

@dataclasses.dataclass(frozen=True)
class LocalPaths:
    ## プロジェクトルートの絶対パス
    BASE_DIR: str = os.path.abspath('./')
    ## dataディレクトリまでの絶対パス
    DATA_DIR: str = os.path.join(BASE_DIR, 'Data')

    ### csv ファイルまでのpath
    HORSE_RESULTS_PATH: str = os.path.join(DATA_DIR, 'horse_results.csv')
    PEDS_PATH : str =  os.path.join(DATA_DIR, 'peds.csv')
    RESULTS_PATH: str =  os.path.join(DATA_DIR, 'results.csv')
    RETURN_PATH : str = os.path.join(DATA_DIR, 'return.csv')

