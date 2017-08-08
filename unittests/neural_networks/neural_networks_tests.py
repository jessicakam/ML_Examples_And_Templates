# 2017/07/29

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import neural_networks as nn


class TestNeuralNetworks(TestCase):
    def test_ANN(self):
        ann = nn.ANN()
        instance = 'ann'
        
        ann.importDataset('Churn_Modelling.csv', 3, 13, 13)
        
        ann.encodeCategoricalData()
        
        ann.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        ann.scaleFeatures2()
        
        ann.build()
        self.assertTrue(ann.classifier, instance + '.classifier should be set.')
        
        ann.compileNN(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        ann.fitToTrainingSet(batch_size = 10, nb_epoch = 100)
        
        ann.predictResults()
        self.assertTrue(ann.y_pred, instance + '.y_pred should be set.')
        
        ann.makeConfusionMatrix()
        
        ann.makeNewPrediction(lst_feature_values=[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000])
        self.assertTrue(ann.new_prediction, instance + '.new_prediction should be set.')
        
        ann.evaluate()
        self.assertTrue(ann.mean, instance + '.mean should be set.')
        self.assertTrue(ann.variance, instance + '.variance should be set.')
        
        ann.improve()
        self.assertTrue(ann.best_parameters, instance + '.best_parameters should be set.')
        self.assertTrue(ann.best_accuracy, instance + '.best_accuracy should be set.')
           
        
    def test_CNN(self):
        cnn = nn.CNN()
        instance = 'cnn'
        
        cnn.build()
        
        cnn.compile()
        
        cnn.fitToImages()
        
        cnn.makeNewPrediction()
        
        
    def test_RNN(self):
        rnn = nn.RNN()
        instance = 'rnn'
        
        rnn.importTrainingSet()
        
        rnn.scaleFeatures()
        
        rnn.getInputsAndOutputs()
        self.assertTrue(rnn.X_train.any(), instance + '.X_train should be set.')
        self.assertTrue(rnn.y_train.any(), instance + '.y_train should be set.')
        
        X_before = rnn.X_train
        rnn.reshape()
        X_after = rnn.X_train
        ##self.assertTrue((X_before.shape[0] != X_after[0]) and (X_before.shape[1] != X_before.shape[1]), instance + '.X_train should change after reshaping')
        
        rnn.build()
        self.assertTrue(rnn.regressor, instance + '.regressor should be set.')
        
        rnn.compileNN()
        
        rnn.fitToTrainingSet()
        self.assertTrue(rnn.real_stock_price, instance + '.real_stock_price should be set.')
        self.assertTrue(rnn.predicted_stock_price, instance + '.predicted_stock_price should be set.')
        
        rnn.makePredictions()
        rnn.visualizeResults()
        self.assertTrue(rnn.best_accuracy, instance + '.rmse should be set.')
        
        
if __name__ == '__main__':
    unittest.main()        
        