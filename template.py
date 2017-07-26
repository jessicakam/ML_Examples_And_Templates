"""
Name: Jessica Kam
Date: 2017/07/25
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer, LabelEncoder, OneHotEncoder

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
    
    def encodeCategoricalDataForIndependentVar(self, column_to_encode=0):
        labelencoder_X = LabelEncoder()
        self.X[:, column_to_encode] = labelencoder_X.fit_transform(self.X[:, column_to_encode])
        onehotencoder = OneHotEncoder(categorical_features=[column_to_encode])
        self.X = onehotencoder.fit_transform(self.X).toarray()
    
    def encodeCategoricalDataForDependentVar(self):
        labelencoder_y = LabelEncoder()
        self.y = labelencoder_y.fit_transform(self.y)
    ###
    def avoidTheDummyVariableTrap(self, start_index=1):
        self.X = self.X[:, start_index:]
        
    def splitIntoTrainingAndTestSets(self, test_size=0.2, random_state=0, *args, **kwargs):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, *args, **kwargs) 
        
    def scaleFeatures(self):
        sc_X = StandardScaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test = sc_X.transform(self.X_test)
        sc_y = StandardScaler()
        self.y_train = sc_y.fit_transform(self.y_train)

from sklearn.linear_model import LinearRegression
class Regression(DataPreprocessing):
    def __init__(self):
        pass

class SimpleLinearRegression(Regression):
    def __init__(self):
        self.regressor = None
        self.y_pred = None
    
    def fitToTrainingSet(self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
    
    def predictTestSetResults(self):
        self.y_pred = self.regressor.predict(self.X_test)
        
    def visualizeResults(self, x_for_scatter, y_for_scatter, color1, color2, title, xlabel, ylabel):
        plt.scatter(x_for_scatter, y_for_scatter, color=color1)
        plt.plot(self.X_train, self.regressor.predict(self.X_train), color=color2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
            
    def visualizeTrainingSetResults(self, color1='red', color2='blue', title='Salary vs Exp (training)', xlabel='Years of Exp', ylabel='Salary'):
        self.visualizeResults(self.X_train, self.y_train, color1, color2, title, xlabel, ylabel)
        
    def visualizeTestSetResults(self, color1='red', color2='blue', title='Salary vs Exp (test)', xlabel='Years of Exp', ylabel='Salary'):
        self.visualizeResults(self.X_test, self.y_test, color1, color2, title, xlabel, ylabel)

import statsmodels.formula.api as sm
class MultipleLinearRegression(SimpleLinearRegression):
    def __init__(self):
        pass
    
    def visualizeTrainingSetResults(self):
        pass
    
    def visualizeTestSetResults(self):
        pass
    
    def findOptimalModelWithBackwardElimination(self):
        pass
        ##TODO - see multiple_linear_regression.py

from sklearn.prepreprocessing import PolynomialFeatures
class PolynomialRegression(Regression):
    def __init__(self):
        self.lin_reg = None
        self.lin_reg_2 = None
        self.poly_reg = None

    def fitLinRegToDataset(self):
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.X, self.y)
    
    def fitPolyRegToDataset(self, degree=4):
        self.poly_reg = PolynomialFeatures(degree=degree)
        X_poly = self.poly_reg.fit_transform(self.X)
        self.poly_reg.fit(X_poly, self.y)
        self.lin_reg_2 = LinearRegression()
        self.lin_reg_2.fit(X_poly, self.y)
    
    def visualizeResults(self, color1, title, xlabel, ylabel):
        plt.scatter(self.X, self.y, color1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    def visualizeLinRegResults(self, color1='red', color2='blue', title='Truth or Bluff (LinReg)', xlabel='Position Lvl', ylabel='Salary'):
        plt.plot(self.X, self.lin_reg.predict(self.X), color=color2)
        self.visualizeResults(color1, title, xlabel, ylabel) #need pass in plt?
        
    def visualizePolyRegResults(self, color1='red', color2='blue', title='Truth or Bluff (PolyReg)', xlabel='Position Level', ylabel='Salary', high_resolution=False, **kwargs):
        X_to_plot = self.X
        if kwargs.get('high_resolution') and kwargs.get('granularity'):
            X_grid = np.arange(min(self.X), max(self.X), kwargs.get('granularity'))
            X_grid = X_grid.reshape((len(X_grid), 1))
            X_to_plot = X_grid
        plt.plot(X_to_plot, self.lin_reg_2.predict(self.poly_reg.fit_transform(X_to_plot)), color2)
        self.visualizeResults(color1, title, xlabel, ylabel) #need pass in plt?
       
    def makePredictionWithLinReg(self, new_value=6.5):
        return self.lin_reg.predict(new_value)
        
    def makePredictionWithPolyReg(self, new_value=6.5):
        return self.lin_reg_2.predict(self.poly_reg.fit_transform(new_value))
        
class SVR(Regression): #hmmm inherit from something else?, what does this stand for again
    def __init__(self):
        pass
        
        
##need to double check all regressions
dummy = MultipleLinearRegression()
dummy.importDataset('50_Startups.csv')
#dummy.fillInMissingData()
dummy.encodeCategoricalDataForIndependentVar(3)
#dummy.encodeCategoricalDataForDependentVar()
dummy.avoidTheDummyVariableTrap()
dummy.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
#dummy.scaleFeatures()
dummy.fitToTrainingSet()
dummy.predictTestSetResults()
#dummy.visualizeTrainingSetResults()
#dummy.visualizeTestSetResults()
dummy.findOptimalModelWithBackwardElimination()




