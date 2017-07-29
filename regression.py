"""
Name: Jessica Kam
Date: 2017/07/25
"""
from data_preprocessing import DataPreprocessing

from sklearn.linear_model import LinearRegression
class Regression(DataPreprocessing):
    def __init__(self):
        self.name= None
    
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
        self.regressor = None
        self.y_pred = None
        self.name = 'SimpleLinReg'
    
    def fitToTrainingSet(self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
    
    def predictTestSetResults(self):
        self.y_pred = self.regressor.predict(self.X_test)
            
    def visualizeTrainingSetResults(self, color1, color2, title, xlabel, ylabel):
        super(SimpleLinearRegression, self).visualizeResults(X_to_plot=self.X_train, x_for_scatter=self.X_train, y_for_scatter=self.y_train, color1, color2, title, xlabel, ylabel)
        
    def visualizeTestSetResults(self, color1, color2, title, xlabel, ylabel):
        super(SimpleLinearRegression, self).visualizeResults(X_to_plot=self.X_train, x_for_scatter=self.X_test, y_for_scatter=self.y_test, color1, color2, title, xlabel, ylabel)
        
import statsmodels.formula.api as sm
class MultipleLinearRegression(SimpleLinearRegression):
    def __init__(self):
        self.name = 'MultipleLinReg'
    
    def visualizeTrainingSetResults(self):
        pass

    def visualizeTestSetResults(self):
        pass
    
    def findOptimalModelWithBackwardElimination(self):
        pass
        ##TODO - see multiple_linear_regression.py
        
from sklearn.preprocessing import PolynomialFeatures
class PolynomialRegression(Regression):
    def __init__(self):
        self.lin_reg = None
        self.lin_reg_2 = None
        self.poly_reg = None
        self.name = 'PolyReg'

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
        plt.scatter(self.X, self.y, color1)
        plt.plot(self.X, self.lin_reg.predict(self.X), color2)
        plt.title(self.generateTitle(title))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    #having a bit of trouble with unsupported input type, maybe bc of version?
    def visualizePolyRegResults(self, color1, color2, title, xlabel, ylabel, **kwargs):
        plt.scatter(self.X, self.y, color1)
        X_to_plot = self.X
        if kwargs.get('high_resolution') and kwargs.get('granularity'):
            X_grid = np.arange(min(self.X), max(self.X), kwargs.get('granularity'))
            X_grid = X_grid.reshape((len(X_grid), 1))
            X_to_plot = X_grid
        plt.plot(X_to_plot, self.lin_reg_2.predict(self.poly_reg.fit_transform(X_to_plot)), color2)
        plt.title(self.generateTitle(title))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def predictTestSetResults(self):
        pass
    
    def makePredictionWithLinReg(self, new_value=6.5):
        return self.lin_reg.predict(new_value)
        
    def makePredictionWithPolyReg(self, new_value=6.5):
        return self.lin_reg_2.predict(self.poly_reg.fit_transform(new_value))

from sklearn.svm import SVR
class SupportVectorRegression(Regression):
    def __init__(self):
        self.sc_X = None
        self.sc_y = None
        self.name = 'SVR'
    
    def scaleFeatures(self):
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.X = self.sc_X.fit_transform(self.X)
        self.y = self.sc_y.fit_transform(self.y)
    
    def fitToDataset(self, kernel='rbf'):
        self.regressor = SVR(kernel)
        self.regressor.fit(self.X, self.y)
    
    def makePredictions(self, value_to_predict=6.5):
        self.y_pred = regressor.predict(value_to_predict)
        self.y_pred = self.sc_y.inverse_transform(self.y_pred)
    
    def visualizeResults(self, color1, color2, title, xlabel, ylabel, **kwargs):
        super(SVR, self).visualizeResults(X_to_plot=self.X, x_for_scatter=self.X, y_for_scatter=self.y, color1, color2, title, xlabel, ylabel, **kwargs)
        
class DecisionTreeRegression(Regression):
    def __init__(self):
        self.name = 'DecisionTreeReg'
    
    def fitToDataset(self, **kwargs):
        self.regressor = DecisionTreeRegressor(**kwargs)
        self.regressor.fit(self.X, self.y)
        
    def predictResult(self, value_to_predict):
        return self.regressor.predict(value_to_predict)
        
    def visualizeResults(self, color1, color2, title, xlabel, ylabel, **kwargs):
        super(DecisionTreeRegression, self).visualizeResults(X_to_plot=self.X, x_for_scatter=self.X, y_for_scatter=self.y, color1, color2, title, xlabel, ylabel, **kwargs)


class RandomForestRegression(DecisionTreeRegression):
    def __init__(self):
        self.name = 'RandomForestReg'
        
    def fitToDataset(self, **kwargs):
        self.regressor = RandomForestRegressor(**kwargs)
        self.regressor.fit(self.X, self.y)
        




