# 2017/07/27

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import regression as reg

class TestRegressionClasses(TestCase):
    
    def test_SimpleLinearRegression(self):
        simple_lin_reg = reg.SimpleLinearRegression()
        instance = 'simple_lin_reg'
        
        simple_lin_reg.importDataset1('Salary_Data.csv')
        self.assertTrue(simple_lin_reg.X.all(), instance + '.X should be set after importing dataset')
        self.assertTrue(simple_lin_reg.y.any(), instance + '.y should be set after importing dataset.')
        
        simple_lin_reg.splitIntoTrainingAndTestSets(test_size=1/3, random_state=0)
        self.assertTrue(simple_lin_reg.X_train.all(), instance + '.X_train should be set after importing dataset')
        self.assertTrue(simple_lin_reg.X_test.all(), instance + 'X_test should be set after importing dataset.')
        self.assertTrue(simple_lin_reg.y_train.any(), instance + '.y_train should be set after importing dataset')
        self.assertTrue(simple_lin_reg.y_test.any(), instance + '.y_test should be set after importing dataset.')
        
        simple_lin_reg.fitToTrainingSet()
        self.assertTrue(simple_lin_reg.regressor, instance + '.regressor has not been set. Training set not fitted.')
        
        simple_lin_reg.predictTestSetResults()
        self.assertTrue(simple_lin_reg.y_pred.any(), instance + '.y_pred has not been set. Results not predicted.')
        
        simple_lin_reg.visualizeTrainingSetResults('red', 'blue', 'Salary vs Exp (training)', 'Years of Exp', 'Salary')
        simple_lin_reg.visualizeTestSetResults('red', 'blue', 'Salary vs Exp (training)', 'Years of Exp', 'Salary')

    def test_MultipleLinearRegression(self):
        multiple_lin_reg = reg.MultipleLinearRegression()
        instance = 'multiple_lin_reg'
        
        multiple_lin_reg.importDataset1('50_Startups.csv')
        
        multiple_lin_reg.encodeCategoricalDataForIndependentVar(column_to_encode=3)
        self.assertTrue(multiple_lin_reg.labelencoder_X, instance + '.labelencoder_X has not been set.')
        self.assertTrue(multiple_lin_reg.onehotencoder, instance + '.onehotencoder has not been set.')
        
        X_before = multiple_lin_reg.X
        multiple_lin_reg.avoidTheDummyVariableTrap(start_index=1)
        X_after = multiple_lin_reg.X
        self.assertTrue(X_before.shape[1] != X_after.shape[1], instance + '.X should have different numbers of columns before and after.')
        
        multiple_lin_reg.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        multiple_lin_reg.fitToTrainingSet()
        
        multiple_lin_reg.predictTestSetResults()
        
        ##maybe: multiple_lin_reg.findOptimalModelWithBackwardsElimination()

    def test_PolynomialRegression(self):
        poly_reg = reg.PolynomialRegression()
        instance = 'poly_reg'
        
        poly_reg.importDataset1('Position_Salaries.csv', 1, 2, 2)
        
        poly_reg.fitLinRegToDataset()
        self.assertTrue(poly_reg.lin_reg, instance + '.lin_reg should have been created.')
        
        poly_reg.fitPolyRegToDataset(degree=4)
        self.assertTrue(poly_reg.lin_reg_2, instance + '.lin_reg_2 should have been created.')
        
        poly_reg.visualizeLinRegResults('red', 'blue', 'Truth or Bluff (Lin Reg)', 'Position Level', 'Salary')
        poly_reg.visualizePolyRegResults('red', 'blue', 'Truth or Bluff (Poly Reg)', 'Position Level', 'Salary')
        poly_reg.visualizePolyRegResults(high_resolution=True, granularity=0.1, color1='red', color2='blue', title='Truth or Bluff (Poly Reg)', xlabel='Position Level', ylabel='Salary')
        
        self.assertTrue(poly_reg.makePredictionWithLinReg(6.5))
        self.assertTrue(poly_reg.makePredictionWithPolyReg(6.5))

    def test_SupportVectorRegression(self):
        svr = reg.SupportVectorRegression()
        instance = 'svr'
        
        svr.importDataset1('Position_Salaries.csv', 1, 2, 2)
        
        svr.scaleFeatures3()
        
        svr.fitToDataset(kernel='rbf')
        self.assertTrue(svr.regressor, instance + '.regressor should have been created.')
        
        svr.makePrediction(6.5)
        self.assertTrue(svr.y_pred, instance + ' prediction should have been made.')
        
        svr.visualizeResults('red', 'blue', 'Truth or Bluff', 'Position Level', 'Salary')
        svr.visualizeResults(high_resolution=True, granularity=0.01, color1='red', color2='blue', title='Truth or Bluff', xlabel='Position Level', ylabel='Salary') 
        
    def test_DecisionTreeRegression(self):
        decision_tree_reg = reg.DecisionTreeRegression()
        instance = 'decision_tree_reg'
        
        decision_tree_reg.importDataset1('Position_Salaries.csv', 1, 2, 2)
        
        decision_tree_reg.fitToDataset(random_state=0)
        self.assertTrue(decision_tree_reg.regressor, instance + '.regressor should have been created.')
        
        self.assertTrue(decision_tree_reg.predictResult(6.5), instance + ' prediction was not able to be made.')
        
        decision_tree_reg.visualizeResults(high_resolution=True, granularity=0.01, color1='red', color2='blue', title='Truth or Bluff', xlabel='Position Level', ylabel='Salary') 
        
    def test_RandomForestRegression(self):
        random_forest_reg = reg.RandomForestRegression()
        instance = 'random_forest_reg'
        
        random_forest_reg.importDataset1('Position_Salaries.csv', 1, 2, 2)
        
        random_forest_reg.fitToDataset(n_estimators=10, random_state=0)
        self.assertTrue(random_forest_reg.regressor, instance + '.regressor should have been created.')
        
        self.assertTrue(random_forest_reg.predictResult(6.5), instance + ' prediction was not able to be made.')
        
        random_forest_reg.visualizeResults(high_resolution=True, granularity=0.01, color1='red', color2='blue', title='Truth or Bluff', xlabel='Position Level', ylabel='Salary') 
        
        
if __name__ == '__main__':
    unittest.main()