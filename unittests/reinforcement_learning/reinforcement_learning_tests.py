# 2017/07/29

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import reinforcement_learning as rl


class TestReinforcementLearningClasses(TestCase):
    def test_RandomSelection(self):
        rs = rl.RandomSelection()
        instance = 'rl'
        
        rs.importDataset('Ads_CTR_Optimisation.csv')
        
        rs.implement(N=1000, d=10)
        self.assertTrue(rs.total_reward, instance + '.total_reward should be set.')
        self.assertTrue(rs.ads_selected, instance + '.ads_selected should be added to.')
        
        rs.visualizeResults('Histogram of ads selections', 'Ads', 'Num of times each selected')
        
    def test_UCB(self):
        ucb = rl.UCB()
        instance = 'ucb'
        
        ucb.importDataset('Ads_CTR_Optimisation.csv')
        
        ucb.implement(N=10000, d=10)
        self.assertTrue(ucb.total_reward, instance + '.total_reward should be set.')
        self.assertTrue(ucb.ads_selected, instance + '.ads_selected should be added to.')
        
        ucb.visualizeResults('Histogram of ads selections', 'Ads', 'Num of times each selected')
        
    def test_ThompsonSampling(self):
        ts = rl.ThompsonSampling()
        instance = 'ts'
        
        ts.importDataset1('Ads_CTR_Optimisation.csv')
        
        ts.implement(N=10000, d=10)
        self.assertTrue(ts.total_reward, instance + '.total_reward should be set.')
        self.assertTrue(ts.ads_selected, instance + '.ads_selected should be added to.')
        
        ts.visualizeResults('Histogram of ads selections', 'Ads', 'Num of times each selected')
        

if __name__ == '__main__':
    unittest.main()