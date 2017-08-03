# 2017/07/29

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import clustering as cluster

class TestClusteringClasses(TestCase):
    
    def test_KMeansClustering(self):
        kmeans = cluster.KMeansClustering()
        instance = 'kmeans'
        
        kmeans.importDataset3('Mall_Customers.csv', [3, 4])
        self.assertTrue(kmeans.X.all(), instance + '.X should be set after importing dataset')
        
        kmeans.findOptimalNumberOfClusters(xlabel='Num of clusters', ylabel='WCSS', max_num_clusters=11, init='k-means++', random_state=42)
        
        kmeans.fitToDataset(n_clusters=5, init='k-means++', random_state=42)
        self.assertTrue(kmeans.kmeans, instance + '.kmeans has not been set.')
        self.assertTrue(kmeans.y_cluster.any(), instance + '.y_cluster has not been set. Data not fitted.')
        
        kmeans.visualizeClusters(num_clusters=5, lst_colors=['red', 'blue', 'green', 'cyan', 'magenta'], title='Clusters of customers', xlabel='Annual Income (k$)', ylabel='Spending Score (1-100)')
        
    def test_HierarchialClustering(self):
        hierarchial_clust = cluster.HierarchialClustering()
        instance = 'hierarchial_clust'
        
        hierarchial_clust.importDataset3('Mall_Customers.csv', [3, 4])
        self.assertTrue(hierarchial_clust.X.all(), instance + '.X should be set after importing dataset')
        
        hierarchial_clust.findOptimalNumberOfClusters(xlabel='Customers', ylabel='Euclidean distances', method='ward')
        
        hierarchial_clust.fitToDataset(n_clusters=5, affinity='euclidean', linkage='ward')
        self.assertTrue(hierarchial_clust.hc, instance + '.hc has not been set.')
        self.assertTrue(hierarchial_clust.y_cluster.any(), instance + '.y_cluster has not been set. Data not fitted.')
        
        hierarchial_clust.visualizeClusters(num_clusters=5, lst_colors=['red', 'blue', 'green', 'cyan', 'magenta'], title='Clusters of customers', xlabel='Annual Income (k$)', ylabel='Spending Score (1-100)')
        
    
    