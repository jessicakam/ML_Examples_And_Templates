"""
Name: Jessica Kam
Date: 2017/07/27
"""
import regression as reg
from unittest import TestCase
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 

class TestRegressionClasses(TestCase):
    
    def test_SimpleLinearRegression(self):
        simple_lin_reg = reg.SimpleLinearRegression()
        simple_lin_reg.importDataset('Salary_Data.csv')
        simple_lin_reg.splitIntoTrainingAndTestSets(test_size=1/3, random_state=0)
        simple_lin_reg.fitToTrainingSet()
        simple_lin_reg.predictTestSetResults()
        simple_lin_reg.visualizeTrainingSetResults('red', 'blue', 'Salary vs Exp (training)', 'Years of Exp', 'Salary')
        simple_lin_reg.visualizeTestSetResults('red', 'blue', 'Salary vs Exp (training)', 'Years of Exp', 'Salary')

    def test_MultipleLinearRegression(self):
        multiple_lin_reg = reg.MultipleLinearRegression()
        multiple_lin_reg.importDataset('50_Startups.csv')
        multiple_lin_reg.encodeCategoricalData(column_to_encode=3)
        multiple_lin_reg.avoidTheDummyVariableTrap(start_index=1)
        multiple_lin_reg.splitIntoTrainAndTestSets(test_size=0.2, random_state=0)
        multiple_lin_reg.fitToTrainingSet()
        multiple_lin_reg.predictTestSetResults()
        ##maybe: multiple_lin_reg.findOptimalModelWithBackwardsElimination()


    def test_PolynomialRegression(self):
        poly_reg = reg.PolynomialRegression()
        poly_reg.importDataset('Position_Salaries.csv', 1, 2, 2)
        poly_reg.fitLinRegToDataset()
        poly_reg.fitPolyRegToDataset(degree=4)
        poly_reg.visualizeLinRegResults('red', 'blue', 'Truth or Bluff (Lin Reg)', 'Position Level', 'Salary')
        poly_reg.visualizePolyRegResults('red', 'blue', 'Truth or Bluff (Poly Reg)', 'Position Level', 'Salary')
        poly_reg.visualizePolyRegResults(high_resolution=True, granularity=0.1, 'red', 'blue' 'Truth or Bluff (Poly Reg)', 'Position Level', 'Salary')
        self.assertTrue(poly_reg.makePredictionWithLinReg())
        self.assertTrue(poly_reg.makePredictionWithPolyReg())

    def test_SupportVectorRegression(self):
        svr = reg.SupportVectorRegression()
        svr.importDataset('Position_Salaries.csv', 1, 2, 2)
        svr.scaleFeatures()
        svr.fitToDataset(kernel='rbf')
        svr.makePredictions(6.5)
        svr.visualizeResults('red', 'blue', 'Truth or Bluff', 'Position Level', 'Salary')
        svr.visualizeResults(high_resolution=True, granularity=0.01, 'red', 'blue', 'Truth or Bluff', 'Position Level', 'Salary') 
        
    def test_DecisionTreeRegression(self):
        decision_tree_reg = reg.DecisionTreeRegression()
        decision_tree_reg.importDataset('Position_Salaries.csv', 1, 2, 2)
        decision_tree_reg.fitToDataset(random_state=0)
        self.assertTrue(decision_tree_reg.predictResult(6.5))
        decision_tree_reg.visualizeResults(higher_resolution=True, granularity=0.01, 'red', 'blue', 'Truth or Bluff', 'Position Lvl', 'Salary')
    
    def test_RandomForestRegression(self):
        random_forest_reg = reg.RandomForestRegression()
        random_forest_reg.importDataset('Position_Salaries.csv', 1, 2, 2)
        random_forest_reg.fitToDataset(n_estimators=10, random_state=0)
        self.assertTrue(random_forest_reg.predictResult(6.5))
        random_forest_reg.visualizeResults(higher_resolution=True, granularity=0.01, 'red', 'blue', 'Truth or Bluff', 'Position Lvl', 'Salary')
        
if __name__ == '__main__':
    unittest.main()