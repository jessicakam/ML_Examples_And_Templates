# 2017/07/25

from data_preprocessing import DataPreProcessing
from data_postprocessing import DataPostProcessing
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
class Regression(DataPreProcessing, DataPostProcessing):
    def __init__(self):
        super(Regression, self).__init__()
    
    def visualizeResults(self, X_to_plot, x_for_scatter, y_for_scatter, color1, color2, title, xlabel, ylabel, **kwargs):
        if kwargs.get('high_resolution') and kwargs.get('granularity'):
            X_grid = np.arange(min(X_to_plot), max(X_to_plot), kwargs.get('granularity'))
            X_grid = X_grid.reshape((len(X_grid), 1))
            X_to_plot = X_grid
        plt.scatter(x_for_scatter, y_for_scatter, color=color1)
        plt.plot(X_to_plot, self.regressor.predict(X_to_plot), color=color2)
        plt.title(self.generateTitle(title))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def generateTitle(self, title):
        return title + ' ' + '(' + self.name + ')'

        
class SimpleLinearRegression(Regression):
    def __init__(self):
        super(Regression, self).__init__()
        self.regressor = None
        self.y_pred = None
        self.name = 'SimpleLinearRegression'
    
    def fitToTrainingSet(self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
    
    def predictTestSetResults(self):
        self.y_pred = self.regressor.predict(self.X_test)
            
    def visualizeTrainingSetResults(self, color1, color2, title, xlabel, ylabel):
        super(SimpleLinearRegression, self).visualizeResults(self.X_train, self.X_train, self.y_train, color1, color2, title, xlabel, ylabel)
        
    def visualizeTestSetResults(self, color1, color2, title, xlabel, ylabel):
        super(SimpleLinearRegression, self).visualizeResults(self.X_train, self.X_test, self.y_test, color1, color2, title, xlabel, ylabel)
  

class MultipleLinearRegression(SimpleLinearRegression):
    def __init__(self):
        self.name = 'MultipleLinearRegression'
    
    def visualizeTrainingSetResults(self):
        pass

    def visualizeTestSetResults(self):
        pass
    
    def findOptimalModelWithBackwardElimination(self):
        pass
        ##maybe later
   
        
from sklearn.preprocessing import PolynomialFeatures
class PolynomialRegression(Regression):
    def __init__(self):
        super(PolynomialRegression, self).__init__()
        self.lin_reg = None
        self.lin_reg_2 = None
        self.poly_reg = None
        self.name = 'PolynomialRegression'

    def fitToTrainingSet(self):
        pass
    
    def fitLinRegToDataset(self):
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.X, self.y)
    
    def fitPolyRegToDataset(self, degree):
        self.poly_reg = PolynomialFeatures(degree=degree)
        X_poly = self.poly_reg.fit_transform(self.X)
        self.poly_reg.fit(X_poly, self.y)
        self.lin_reg_2 = LinearRegression()
        self.lin_reg_2.fit(X_poly, self.y)
    
    def visualizeResults(self, X_to_plot, x_for_scatter, y_for_scatter, color1, color2, title, xlabel, ylabel):
        pass

    #having a bit of trouble with unsupported input type
    def visualizeLinRegResults(self, color1, color2, title, xlabel, ylabel):
        plt.scatter(self.X, self.y, color = color1)
        plt.plot(self.X, self.lin_reg.predict(self.X), color = color2)
        plt.title(self.generateTitle(title))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    #having a bit of trouble with unsupported input type, maybe bc of version?
    def visualizePolyRegResults(self, color1, color2, title, xlabel, ylabel, **kwargs):
        plt.scatter(self.X, self.y, color=color1)
        X_to_plot = self.X
        if kwargs.get('high_resolution') and kwargs.get('granularity'):
            X_grid = np.arange(min(self.X), max(self.X), kwargs.get('granularity'))
            X_grid = X_grid.reshape((len(X_grid), 1))
            X_to_plot = X_grid
        plt.plot(X_to_plot, self.lin_reg_2.predict(self.poly_reg.fit_transform(X_to_plot)), color=color2)
        plt.title(self.generateTitle(title))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def predictTestSetResults(self):
        pass
    
    def makePredictionWithLinReg(self, new_value):
        return self.lin_reg.predict(new_value)
        
    def makePredictionWithPolyReg(self, new_value):
        return self.lin_reg_2.predict(self.poly_reg.fit_transform(new_value))

        
from sklearn.svm import SVR
class SupportVectorRegression(Regression):
    def __init__(self):
        super(SupportVectorRegression, self).__init__()
        self.sc_X = None
        self.sc_y = None
        self.name = 'SVR'
    
    def fitToDataset(self, kernel='rbf'):
        self.regressor = SVR(kernel)
        self.regressor.fit(self.X, self.y)
    
    def makePrediction(self, value_to_predict):
        self.y_pred = self.regressor.predict(value_to_predict)
        self.y_pred = self.sc_y.inverse_transform(self.y_pred)
    
    def visualizeResults(self, color1, color2, title, xlabel, ylabel, **kwargs):
        super(SupportVectorRegression, self).visualizeResults(self.X, self.X, self.y, color1, color2, title, xlabel, ylabel, **kwargs)

        
from sklearn.tree import DecisionTreeRegressor      
class DecisionTreeRegression(Regression):
    def __init__(self):
        super(DecisionTreeRegression, self).__init__()
        self.name = 'DecisionTreeRegression'
    
    def fitToDataset(self, **kwargs):
        self.regressor = DecisionTreeRegressor(**kwargs)
        self.regressor.fit(self.X, self.y)
        
    def predictResult(self, value_to_predict):
        return self.regressor.predict(value_to_predict)
        
    def visualizeResults(self, color1, color2, title, xlabel, ylabel, **kwargs):
        super(DecisionTreeRegression, self).visualizeResults(self.X, self.X, self.y, color1, color2, title, xlabel, ylabel, **kwargs)


from sklearn.ensemble import RandomForestRegressor
class RandomForestRegression(DecisionTreeRegression):
    def __init__(self):
        super(RandomForestRegression, self).__init__()
        self.name = 'RandomForestRegression'
        
    def fitToDataset(self, **kwargs):
        self.regressor = RandomForestRegressor(**kwargs)
        self.regressor.fit(self.X, self.y)
        




