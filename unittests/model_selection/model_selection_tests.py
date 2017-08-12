# 2017/08/06

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import model_selection

class TestModelSelection(TestCase):
    def test_ModelSelection(self):
        ms = model_selection.ModelSelection()
        instance = 'ms'
        
        ms.importDataset2('Social_Network_Ads.csv', [2, 3], 4)
        
        ms.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        
        ms.scaleFeatures2()
        
        ms.fitToTrainingSet(kernel='rbf', random_state=0)
        self.assertTrue(ms.classifier, instance + '.classifier has not been created.')
        
        ms.predictResults()
        
        ms.makeConfusionMatrix()
        
        ms.applyKFoldCrossValidation(estimator = ms.classifier, X = ms.X_train, y = ms.y_train, cv=10)
        self.assertTrue(ms.accuracies.any(), instance + '.accuracies has not been set.')
        self.assertTrue(ms.mean.any(), instance + '.mean has not been set.')
        self.assertTrue(ms.std.any(), instance + '.std has not been set.')
        
        ms.applyGridSearchToFindBestModels()
        self.assertTrue(ms.grid_search, instance + '.grid_search has not been set.')
        self.assertTrue(ms.best_accuracy, instance + '.best_accuracy has not been set.')
        self.assertTrue(ms.best_parameters, instance + '.best_parameters has not been set.')
        
        ms.visualizeTrainingSetResults('Kernel SVM (Training set)', 'Age', 'Estimated Salary')
        ms.visualizeTestSetResults('Kernel SVM (Test set)', 'Age', 'Estimated Salary')

    # Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#
    def test_XGBoost(self):
        xgb = model_selection.XGBoost()
        instance = 'xgb'
        
        xgb.importDataset1('Churn_Modelling.csv', 3, 13, 13)
        
        xgb.encodeCategoricalData()
        self.assertTrue(xgb.onehotencoder, instance + '.onehotencoder has not been created.')
        
        xgb.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        xgb.fitToTrainingSet()
        self.assertTrue(xgb.classifier, instance + '.classifier has not been created.')
        
        xgb.predictResults()
        
        xgb.makeConfusionMatrix()
        
        xgb.applyKFoldCrossValidation(estimator=xgb.classifier, X=xgb.X_train, y=xgb.y_train, cv=10)    

if __name__ == '__main__':
    unittest.main()