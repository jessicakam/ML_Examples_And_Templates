"""
2017/07/29
"""

from data_preprocessing import DataPreprocessing

class AssociationRuleLearning(DataPreprocessing):
    pass

from apyori import apriori
class Apriori(AssociationRuleLearning):
    def __init__(self):
        pass
    
    def preprocessData(self, csv_file, header, num_observations, range_max):
        dataset = pd.read_csv(csv_file, header)
        self.transactions = []
        for i in range(0, num_observations):
            self.transactions.append([str(dataset.values[i,j]) for j in range(0, range_max)])
    
    def trainOnDataset(self, **kwargs):
        self.rules = apriori(self.transactions, **kwargs)
    
    def visualizeResults(self):
        self.results = list(self.rules)

#later
#class Eiat(DataPreprocessing): #correct spelling?
#    pass
