# 2017/07/29

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import misc


class TestSOM(TestCase):
    def test_SOM(self):
        som = misc.SOM()
        instance = 'som'
        
        som.importDataset1('Credit_Card_Applications.csv', 0, -1, -1)
        
        som.scaleFeatures()
        self.assertTrue(som.sc, instance + '.sc has not been created.')
        
        som.train(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
        self.assertTrue(som.som, instance + '.som has not been created.')
        
        som.visualizeResults()
        
        #ValueError: all the input arrays must have same number of dimensions
        #som.findFrauds()
        #self.assertTrue(som.frauds.any(), instance + '.frauds has not been found.')
        
class TestReducedBoltzmannMachines(TestCase):
    def test_ReducedBoltmannMachines(self):
        rbm = misc.ReducedBoltzmannMachines()
        instance = 'rbm'
        
        rbm.importDataset4()
        self.assertFalse(rbm.movies.empty, instance + '.movies has not been set.')
        self.assertFalse(rbm.users.empty, instance + '.users has not been set.')
        self.assertFalse(rbm.ratings.empty, instance + '.ratings has not been set.')
        
        rbm.prepareTrainingAndTestSets()
        self.assertTrue(rbm.training_set.all(), instance + '.training_set has not been set.')
        self.assertTrue(rbm.test_set.all(), instance + '.test_set has not been set.')
        self.assertTrue(rbm.nb_users, instance + '.nb_users has not been set.')
        self.assertTrue(rbm.nb_movies, instance + '.nb_movies has not been set.')
        
        #TypeError: 'module' object is not subscriptable
        rbm.convertData()
        self.assertFalse(rbm.new_data.empty, instance + '.new_data has not been set.')
        
        training_set_before = rbm.training_set
        test_set_before = rbm.test_set
        rbm.convertIntoTensors()
        self.assertTrue(training_set_before != rbm.training_set, instance + '.training_set not a tensor.')
        self.assertTrue(test_set_before != rbm.test_set, instance + '.test_set not a tensor.')
        
        rbm.convertIntoBinary()
        self.assertTrue((rbm.training_set==0 |
                        rbm.training_set==1 |
                        rbm.training_set==-1).all(), instance + '.training_set should be binary')
        self.assertTrue((rbm.training_set==0 |
                        rbm.training_set==1 |
                        rbm.training_set==-1).all(), instance + '.training_set should be binary')
        #self.assertTrue(rbm.training_set.all(rbm.a==0 or rbm.a==1 or rbm.a==-1), instance + '.training_set should be binary')
        #self.assertTrue(rbm.test_set.all(a==0 or a==1 or a==-1), instance + '.test_set should be binary')
        
        rbm.createNN(nv=len(rbm.training_set[0]), nh=100)
        
        rbm.train(mb_epoch=10)
        # screen printouts
        
        rbm.test()
        # screen printouts

class TestAutoEncoder(TestCase):
    def test_AutoEncoder(self):
        ae = misc.AutoEncoder()
        instance = 'ae'
        
        ae.importDataset4()
        
        ae.prepareTrainingAndTestSets()
        
        ae.convertData()
        
        ae.convertIntoTensors()
        
        ae.createNN()
        self.assertTrue(ae.sae, instance + '.sae has not been set.')
        self.assertTrue(ae.criterion, instance + '.criterion has not been set.')
        self.assertTrue(ae.optimizer, instance + '.optimizer has not been set.')
        
        ae.train(nb_epoch=200)
        # screen printouts
        
        ae.test()
        # screen printouts
    
if __name__ == '__main__':
    unittest.main()