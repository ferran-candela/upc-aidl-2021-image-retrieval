import os
import time
import pandas as pd
from datetime import timedelta
from PIL import Image

from config import FoldersConfig

class ProcessTime:
    def __init__(self):
        super().__init__()
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise Exception(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        duration = timedelta(seconds=elapsed_time)
        print(f"Elapsed time: {duration}")
        return str(duration)

    def current_time(self):
        """Get current elapsed time"""
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        duration = timedelta(seconds=elapsed_time)
        return str(duration)

class LogFile:
    def __init__(self, fields):
        super().__init__()
        #exemple: fields = ['ModelName', 'ParametersCount', 'ProcessTime', 'Average']
        self._log_df = pd.DataFrame(columns = fields)

    def writeLogFile(self, values):
        #exemple: values = {'ModelName':'test', 'ParametersCount':87, 'ProcessTime':92, 'Average':87.33}        
        self._log_df = self._log_df.append(values, ignore_index=True)

    def getLogFile(self):
        return self._log_df

    def saveLogFile_to_csv(self, processname):
        #Log directory
        log_dir = FoldersConfig.LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self._log_df.to_csv(log_dir + processname + '_log_' + time.strftime("%Y%m%d-%H%M%S") + '.csv',index=False)

    def printLogFile(self):
        print(self._log_df)


class PrepareData:
    def __init__(self):
        super().__init__()


    def filterPoductFashion(self, labels_file, dataset_base_dir):

        df_dataset = pd.read_csv(labels_file, error_bad_lines=False)

        print(df_dataset.count())
        print(df_dataset.masterCategory.unique())

        different_clothes = ['Bra', 'Kurtas', 'Briefs', 'Sarees', 'Innerwear Vests', 
                            'Kurta Sets', 'Shrug', 'Camisoles', 'Boxers', 'Dupatta', 
                            'Capris', 'Bath Robe', 'Tunics', 'Trunk', 'Baby Dolls', 
                            'Kurtis', 'Suspenders', 'Robe', 'Salwar and Dupatta', 
                            'Patiala', 'Stockings', 'Tights', 'Churidar', 'Shapewear',
                            'Nehru Jackets', 'Salwar', 'Rompers', 'Lehenga Choli',
                            'Clothing Set', 'Belts']

        is_clothes = df_dataset['masterCategory'] == 'Apparel'
        is_shoes = df_dataset['masterCategory'] == 'Footwear'
        is_differenet_clothes = df_dataset['articleType'].isin(different_clothes)

        df_clothes_shoes = df_dataset[(is_clothes | is_shoes) & ~is_differenet_clothes]

        print(df_clothes_shoes.count())
        print(df_clothes_shoes.articleType.unique().size)

        df_clothes_shoes.to_csv(os.path.join(dataset_base_dir, "clothes_shoes.csv"),index=False)



def ImageSize(PILimg):
    width, height = PILimg.size
    return str(width) + "x" + str(height)

