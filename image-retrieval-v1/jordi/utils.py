import time
import pandas as pd
from datetime import timedelta
from PIL import Image

class ProcessTime:
    def __init__(self):
        super().__init__()
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        duration = timedelta(seconds=elapsed_time)
        print(f"Elapsed time: {duration}")
        return str(duration)

class LogFile:
    def __init__(self,fields):
        super().__init__()
        #exemple: fields = ['ModelName', 'ParametersCount', 'ProcessTime', 'Average']
        self._log_df = pd.DataFrame(columns = fields)

    def writeLogFile(self,values):
        #exemple: values = {'ModelName':'test', 'ParametersCount':87, 'ProcessTime':92, 'Average':87.33}        
        self._log_df = self._log_df.append(values, ignore_index=True)

    def getLogFile(self):
        return self._log_df

    def saveLogFile_to_csv(self,config):
        self._log_df.to_csv(config["log_dir"] + 'log_' + time.strftime("%Y%m%d-%H%M%S") + '.csv',index=False)

    def printLogFile(self):
        print(self._log_df)

def ImageSize(PILimg):
    width, height = PILimg.size
    return str(width) + "x" + str(height)

