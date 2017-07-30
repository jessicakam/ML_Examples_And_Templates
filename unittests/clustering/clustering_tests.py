"""
Date: 2017/07/29
"""
import clustering as cluster
from unittest import TestCase
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 

class TestClusteringClasses(TestCase):
    
    def test_KMeans(self):
        kmeans = cluster.KMeans()
        kmeans.importDataset('Mail_Customers.csv', list_of_columns=[3, 4])
        kmeans.findOptimalNumberOfClusters(max_num_clusters=11, init='k-means++', random_state=42)
        kmeans.fitToDataset(n_clusters=5, init='k-means++', random_state=42)
        kmeans.visualizeResults(num_clusters=5, lst_colors=['red', 'blue', 'green', 'cyan', 'magenta'], 'Clusters of customers', 'Annual Income (k$)', 'Spending Score (1-100)') #
        
    def test_HierarchialClustering(self):
        hierarchial_clust = cluster.HierarchialClustering()
        hierarchial_clust.importDataset('Mail_Customers.csv', list_of_columns=[3, 4])
        hierarchial_clust.findOptimalNumberOfClusters(method='ward')
        hierarchial_clust.fitToDataset(n_clusters=5, affinity='euclidean', linkage='ward')
        hierarchial_clust.visualizeResults(num_clusters=5, lst_colors=['red', 'blue', 'green', 'cyan', 'magenta'], 'Clusters of customers', 'Annual Income (k$)', 'Spending Score (1-100)') #
        
    
    