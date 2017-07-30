#Date: 2017/07/29

import neural_networks as nn
from unittest import TestCase
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 

class TestRegressionClasses(TestCase):
    def test_ANN(self):
        ann = nn.ANN()
        ann.importDataset('Churn_Modelling.csv', 3, 13, 13)
        ann.encodeCategoricalData(1) ##ok?
        ann.encodeCategoricalData(2) ##ok???
        ann.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        ann.scaleFeatures()
        ann.build()
        ann.compileNN(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        ann.fitToTrainingSet(batch_size = 10, nb_epoch = 100)
        ann.predictResults()
        ann.makeConfusionMatrix()
        ann.makeNewPrediction(lst_feature_values=[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000])
        ann.evaluate()
        ann.improve()
        
        
        
    def test_CNN(self):
        cnn = nn.CNN()
        cnn.convolution(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')
        cnn.maxPooling(pool_size = (2, 2))
        cnn.addConvolutionLayer() ###
        cnn.flatten()
        cnn.fullConnection() ##
        cnn.build()
        cnn.compile()
        cnn.fitToImages() ##
        cnn.makeNewPrediction()
        
    def test_RNN(self):
        rnn = nn.RNN()
        rnn.importTrainingSet()
        rnn.scaleFeatures()
        rnn.getInputsAndOutputs()
        rnn.reshape()
        rnn.build()
        rnn.compileNN()
        rnn.fitToTrainingSet()
        rnn.makePredictions()
        rnn.visualizeResults()
        
        
        
        