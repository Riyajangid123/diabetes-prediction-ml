import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin

class OutlierCapping(BaseEstimator,TransformerMixin):
        def fit(self,x,y=None):
            x=np.array(x)
            self.q1=np.percentile(x,25,axis=0)
            self.q3=np.percentile(x,75,axis=0)
            self.IQR=self.q3-self.q1
            return self
        def transform(self,x):
            x=np.array(x)
            lower=self.q1-1.5*self.IQR
            upper=self.q3+1.5*self.IQR
            return np.clip(x,lower,upper)

class DataPreprocessing:
    def preprocess_data(self,data):
        x=data.drop('Outcome',axis=1)
        y=data['Outcome']

        num_col=x.select_dtypes(exclude='object').columns
        cat_col=x.select_dtypes(include='object').columns

        pipeline=Pipeline(steps=[('Imputer',SimpleImputer(strategy='median')),
                                 ('Outlier',OutlierCapping()),
                                 ('Scaler',StandardScaler())])
        transformer=ColumnTransformer([('pipeline',pipeline,num_col)])

        return x,y,transformer