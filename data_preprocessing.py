# 2017/07/25

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer, LabelEncoder, OneHotEncoder

class DataPreProcessing():
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
    
    def importDataset1(self, csv_file, X_start_index=0, X_end_index=-1, y_index=-1):
        dataset = pd.read_csv(csv_file)
        self.X = dataset.iloc[:, X_start_index:X_end_index].values
        self.y = dataset.iloc[:, y_index].values
    
    def importDataset2(self, csv_file, lst_columns, y_index):
        dataset = pd.read_csv(csv_file)
        self.X = dataset.iloc[:, lst_columns].values
        self.y = dataset.iloc[:, y_index].values
        
    def importDataset3(self, csv_file, list_of_columns):
        dataset = pd.read_csv(csv_file)
        self.X = dataset.iloc[:, list_of_columns].values
        
    def fillInMissingData(self, filler='NaN', strategy='mean', axis=0, index_start_fill=1, index_end_fill=3, **kwargs):
        imputer = Imputer(missing_values=filler, strategy=strategy, axis=axis, **kwargs)
        imputer = imputer.fit(self.X[:, index_start_fill:index_end_fill])
        self.X[:, index_start_fill: index_end_fill] = imputer.transform(self.X[:, index_start_fill:index_end_fill])
    
    def encodeCategoricalDataForIndependentVar(self, column_to_encode=0):
        self.labelencoder_X = LabelEncoder()
        self.X[:, column_to_encode] = self.labelencoder_X.fit_transform(self.X[:, column_to_encode])
        self.onehotencoder = OneHotEncoder(categorical_features=[column_to_encode])
        self.X = self.onehotencoder.fit_transform(self.X).toarray()
    
    def encodeCategoricalDataForDependentVar(self):
        labelencoder_y = LabelEncoder()
        self.y = labelencoder_y.fit_transform(self.y)
    
    def avoidTheDummyVariableTrap(self, start_index=1):
        self.X = self.X[:, start_index:]
        
    def splitIntoTrainingAndTestSets(self, **kwargs):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, **kwargs) 
        
    def scaleFeatures1(self):
        self.sc_X = StandardScaler()
        self.X_train = self.sc_X.fit_transform(self.X_train)
        self.X_test = self.sc_X.transform(self.X_test)
        self.sc_y = StandardScaler()
        self.y_train = self.sc_y.fit_transform(self.y_train)

    def scaleFeatures2(self):
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)

    def scaleFeatures3(self):
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.X = self.sc_X.fit_transform(self.X)
        self.y = self.sc_y.fit_transform(self.y)