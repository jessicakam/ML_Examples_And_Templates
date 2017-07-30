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
        ann.initialize()
        ann.addInputLayerAndFirstHidden(output_dim=6, init='uniform', activation='relu', input_dim=11)
        ann.addHiddenLayer(output_dim=6, init='uniform', activation='relu')
        ann.addOutputLayer(output_dim = 1, init = 'uniform', activation = 'sigmoid')
        ann.compileNN(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        ann.fitToTrainingSet(batch_size = 10, nb_epoch = 100)
        ann.predictResults()
        ann.makeConfusionMatrix()
        
        
    def test_CNN(self):
        cnn = nn.CNN()
        cnn.convolution(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')
        cnn.maxPooling(pool_size = (2, 2))
        cnn.addConvolutionLayer() ###
        cnn.flatten()
        cnn.fullConnection() ##
        cnn.compileNN()
        cnn.fitToImages() ##
        
        
    def test_RNN(self):
        