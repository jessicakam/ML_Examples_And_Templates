"""
Name: Jessica Kam
Date: 2017/07/25
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer

class DataPreprocessing():
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def importDataset(self, dataset_name='Data.csv', X_start_index=0, X_end_index=-1, y_index=-1):
        dataset = pd.read_csv(dataset_name)
        self.X = dataset.iloc[:, X_start_index:X_end_index].values
        self.y = dataset.iloc[:, y_index].values
    
    def fillInMissingData(self, filler='NaN', strategy='mean', axis=0, index_start_fill=1, index_end_fill=3, *args, **kwargs):
        imputer = Imputer(missing_values=filler, strategy=strategy, axis=axis, *args, **kwargs)
        imputer = imputer.fit(self.X[:, index_start_fill:index_end_fill])
        self.X[:, index_start_fill: index_end_fill] = imputer.transform(self.X[:, index_start_fill:index_end_fill])
    
    def splitIntoTrainingAndTestSet(self, test_size=0.2, random_state=0, *args, **kwargs):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, *args, **kwargs) 
        
    def featureScaling(self):
        sc_X = StandardScaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test = sc_X.transform(self.X_test)
        sc_y = StandardScaler()
        self.y_train = sc_y.fit_transform(self.y_train)
        
from sklearn.linear_model import LinearRegression
class SimpleLinearRegression(DataPreprocessing):
    def __init__(self):
        self.regressor = None
        self.y_pred = None

    def featureScaling(self):
        pass
    
    def fitToTrainingSet(self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
    
    def predictingTestSetResults(self):
        self.y_pred = self.regressor.predict(self.X_test)
        
    def visualizeTrainingSetResults(self, color1='red', color2='blue', title='Salary vs Exp (training)', xlabel='Years of Exp', ylabel='Salary'):
        plt.scatter(self.X_test, self.y_test, color=color1)
        plt.plot(self.X_train, self.regressor.predict(self.X_train), color=color2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylavel(ylabel)
        plt.show()
        
    def visualizeTestSetResults(self, color1='red', color2='blue', title='Salary vs Exp (test)', xlabel='Years of Exp', ylabel='Salary'):
        plt.scatter(self.X_test, self.y_test, color=color1)
        plt.plot(self.X_train, self.regressor.predict(self.X_train), color=color2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        
        
        
        
        
        
        
        
        
        
        
        