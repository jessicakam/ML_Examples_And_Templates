# 2017/07/29

from data_preprocessing import DataPreProcessing
from data_postprocessing import DataPostProcessing
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

class NLP(DataPreProcessing, DataPostProcessing):
    def importDataset(self, file, **kwargs):
        self.dataset = pd.read_csv(file, **kwargs)
        
    def cleanTexts(self, max_range):
        nltk.download('stopwords')
        self.corpus = []
        for i in range(0, max_range):
            review = re.sub('[^a-zA-Z]', ' ', self.dataset['Review'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            self.corpus.append(review)

    def createBagOfWordsModel(self, **kwargs):
        cv = CountVectorizer(**kwargs)
        self.X = cv.fit_transform(self.corpus).toarray()
        self.y = self.dataset.iloc[:, 1].values
    
    def fitNaiveBayesToTrainingSet(self):
        self.classifier = GaussianNB()
        self.classifier.fit(self.X_train, self.y_train)

