#Date: 2017/07/29

import reinforcement_learning as rl
from unittest import TestCase
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 

class TestReinforcementLearningClasses(TestCase):
    def test_randomSampling(self):
        rs = rl.RandomSampling()
        rs.importDataset('Ads_CTR_Optimization.csv')
        rs.implement()
        rs.visualizeResults('Histogram of ads selections', 'Ads', 'Num of times each selected')
        
    def test_UCB(self):
        ucb = rl.UCB()
        ucb.importDataset('Ads_CTR_Optimization.csv')
        ucb.implement()
        ucb.visualizeResults('Histogram of ads selections', 'Ads', 'Num of times each selected')
        
    def test_ThompsonSampling(self):
        ts = rl.ThompsonSampling()
        ts.importDataset('Ads_CTR_Optimization.csv')
        ts.implement()
        ts.visualizeResults('Histogram of ads selections', 'Ads', 'Num of times each selected')
        