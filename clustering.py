"""
Name: Jessica Kam
Date: 2017/07/29
"""
from data_preprocessing import DataPreprocessing

class Cluster(DataPreprocessing):
    def __init__(self):
        pass
    
    def importDataset(self, csv_file, list_of_columns):
        dataset = pd.read_csv(csv_file)
        self.X = dataset.iloc[:, list_of_columns].values
        
    def visualizeClusters(self, num_clusters, lst_colors, title, xlabel, ylabel):
        for i in range num_clusters:
            plt.scatter(self.X[self.y_cluster == i, 0], self.X[self.y_cluster == i, 1], s=100, c=lst_colors[i], label='Cluster'+i+1)
        if 'KMeans' in self.name:
            plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
        
from sklearn.cluster import KMeans
class KMeans(Cluster):
    def __init__(self):
        self.name = 'KMeans'
        
    def findOptimalNumberOfClusters(self, max_num_clusters=11, **kwargs):
        wcss = []
        for i in range(1, max_num_clusters):
            kmeans = KMeans(n_clusters = i, **kwargs)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, max_num_clusters), wcss)
        plt.title('Opt Num Clusters Using the Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    def fitToDataset(self, **kwargs):
        self.kmeans = KMeans(**kwargs)
        self.y_cluster = self.kmeans.fit_predict(self.X)
                
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
class HierarchialClustering(Cluster):
    def findOptimalNumberOfClusters(self, **kwargs):
        dendrogram = sch.dendrogram(sch.linkage(self.X, **kwargs))
        plt.title('Opt Num Clusters Using Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()
        
    def fitToDataset(self, **kwargs):
        self.hc = AgglomerativeClustering(**kwargs)
        self.y_cluster = self.hc.fit_predict(self.X)
