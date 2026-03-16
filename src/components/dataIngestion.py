import os
import pandas as pd

class DataIngestion:
    def data_ingest(self,path):
        data=pd.read_csv(path)
        return data