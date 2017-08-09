# 2017/08/06

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import model_selection as ms

class TestModelSelection(TestCase):
    def test_ModelSelection(self):
gs = ms.GridSearch()
        instance = 'gs'
        
        gs.importDataset2('Social_Network_Ads.csv', [2, 3], 4)
        
        gs.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        
        gs.scaleFeatures2()
        
        gs.fitToTrainingSet(kernel='rbf', random_state=0)
        self.assertTrue(gs.classifier, instance + '.classifier has not been created.')
        
        gs.predictResults()
        
        gs.makeConfusionMatrix()
        
        gs.applyKFoldCrossValidation(cv=10)
        self.assertTrue(gs.accuracies, instance + '.accuracies has not been set.')
        self.assertTrue(gs.mean, instance + '.mean has not been set.')
        self.assertTrue(gs.std, instance + '.std has not been set.')
        
        gs.applyGridSearchToFindBestModels()
        self.assertTrue(gs.grid_search, instance + '.grid_search has not been set.')
        self.assertTrue(gs.best_accuracy, instance + '.best_accuracy has not been set.')
        self.assertTrue(gs.best_parameters, instance + '.best_parameters has not been set.')
        
        gs.visualizeTrainingSetResults('Kernel SVM (Test set)', 'Age', 'Estimated Salary')
        gs.visualizeTestSetResults('Kernel SVM (Test set)', 'Age', 'Estimated Salary')

    # Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#
    def test_XGBoost(self):
        xgb = ms.XGBoost()
        instance = 'xgb'
        
        xgb.importDataset1('Churn_Modelling.csv', 3, 13, 13)
        
        xgb.encodeCategoricalData()
        self.assertTrue(xgb.onehotencoder, instance + '.onehotencoder has not been created.')
        
        xgb.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        xgb.fitToTrainingSet()
        self.assertTrue(xgb.classifier, instance + '.classifier has not been created.')
        
        xgb.predictResults()
        
        xgb.makeConfusionMatrix()
        
        xgb.applyKFoldCrossValidation(estimator=self.classifier, X=self.X_train, y=self.y_train, cv=10)
        self.assertTrue(xgb.accuracies, instance + '.accuracies has not been set.')
        self.assertTrue(xgb.mean, instance + '.mean has not been set.')
        self.assertTrue(xgb.std, instance + '.std has not been set.')
    

if __name__ == '__main__':
    unittest.main()