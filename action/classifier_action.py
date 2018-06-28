import pickle
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from resources import document
import pandas as pd
import operator
import logging
import warnings

labelMap={}

class classifier_action:

    logger=logging.getLogger(__name__)
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def getlabelfrommodel(self,text):
        try:
            tfidffile = open('TFIDF.pkl', 'rb')
            tfidf_transformer = TfidfTransformer()
            loaded_vec = CountVectorizer(vocabulary=pickle.load(tfidffile))
            test = pd.DataFrame()
            test['doc'] = [text]
            test_tfidf = tfidf_transformer.fit_transform(loaded_vec.transform(test['doc']))
            loaded_model = pickle.load(open('RandomForest.pkl', 'rb'))
            label = loaded_model.predict(test_tfidf)[0]
            probabs = loaded_model.predict_proba(test_tfidf)
            confidence_scores={}
            for i, p in enumerate(probabs[0]):
                if (p > 0):
                    confidence_scores[labelMap[str(i)]]=p*100
            ordered_confidence=sorted(confidence_scores.items(),key=operator.itemgetter(1),reverse=True)
            doc=document.document(labelMap[str(label)],ordered_confidence)
            return doc
        except FutureWarning as w:
            self.logger.warning(exc_info=True)
        except Exception as e:
            self.logger.error('Failed to classify using the model ', exc_info=True)

    def getLabelMappings(self):
        mappingsfile=open('LabelMapping.txt','r')
        lines=mappingsfile.readlines()
        for line in lines:
            keyval=line.split(':')
            labelMap[keyval[0]]=keyval[1].strip()
            self.logger.info('Encoded label map for the categories is loaded')
