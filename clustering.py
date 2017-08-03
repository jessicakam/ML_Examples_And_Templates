# 2017/07/29

from data_preprocessing import DataPreProcessing
from data_postprocessing import DataPostProcessing
import matplotlib.pyplot as plt

class Cluster(DataPreProcessing, DataPostProcessing):
    def __init__(self):
        super(Cluster, self).__init__()
        
    def visualizeClusters(self, num_clusters, lst_colors, title, xlabel, ylabel):
        for i in range(num_clusters):
            plt.scatter(self.X[self.y_cluster == i, 0], self.X[self.y_cluster == i, 1], s=100, c=lst_colors[i], label='Cluster'+str(i+1))
        if 'KMeans' in self.name:
            plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
        
from sklearn.cluster import KMeans
class KMeansClustering(Cluster):
    def __init__(self):
        super(KMeansClustering, self).__init__()
        self.name = 'KMeansClustering'
        
    def findOptimalNumberOfClusters(self, xlabel, ylabel, max_num_clusters, **kwargs):
        wcss = []
        for i in range(1, max_num_clusters):
            kmeans = KMeans(n_clusters = i, **kwargs)
            kmeans.fit(self.X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, max_num_clusters), wcss)
        plt.title('Opt Num Clusters Using the Elbow Method')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def fitToDataset(self, **kwargs):
        self.kmeans = KMeans(**kwargs)
        self.y_cluster = self.kmeans.fit_predict(self.X)
                
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
class HierarchialClustering(Cluster):
    def __init__(self):
        super(HierarchialClustering, self).__init__()
        self.name = 'HierarchialClustering'
        
    def findOptimalNumberOfClusters(self, xlabel, ylabel, **kwargs):
        self.dendrogram = sch.dendrogram(sch.linkage(self.X, **kwargs))
        plt.title('Opt Num Clusters Using Dendrogram')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    def fitToDataset(self, **kwargs):
        self.hc = AgglomerativeClustering(**kwargs)
        self.y_cluster = self.hc.fit_predict(self.X)
