# import pickle
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.externals import joblib
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from resources import document
import pandas as pd
import operator

labelMap={}

BUCKET_NAME = 'mydocclassifierapp'
VECTOR_FILE_NAME = 'TFIDF.pkl'
MODEL_FILE_NAME = 'RandomForest.pkl'
MAPPING_FILE_NAME = 'LabelMapping.txt'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME
VECTOR_LOCAL_PATH = '/tmp/' + VECTOR_FILE_NAME
MAPPING_LOCAL_PATH = '/tmp/' + MAPPING_FILE_NAME
conn = S3Connection()
bucket = conn.get_bucket(BUCKET_NAME)
key_obj = Key(bucket)
class classifier_action:

    def load_model(self):
        try:
            key_obj.key = MODEL_FILE_NAME
            key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
            return joblib.load(MODEL_LOCAL_PATH)
        except Exception as e:
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Problem loading trained model file^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print(e)

    def load_vector(self):
        try:
            key_obj.key = VECTOR_FILE_NAME
            key_obj.get_contents_to_filename(VECTOR_LOCAL_PATH)
            return joblib.load(VECTOR_LOCAL_PATH)
        except Exception as e:
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Problem loading TFIDF vector(dimensions from training)^^^^^^^^^^^^^^^^^^^^^^^")
            print(e)

    def getlabelfrommodel(self,text):
        tfidf_transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(vocabulary=self.load_vector())
        print("***********************************loaded transformer*************************************")
        test = pd.DataFrame()
        test['doc'] = [text]
        test_tfidf = tfidf_transformer.fit_transform(loaded_vec.transform(test['doc']))
        print("***********************************transformed test input*************************************")
        loaded_model = self.load_model()
        print("***********************************loaded model*************************************")
        label = loaded_model.predict(test_tfidf)[0]
        print("***********************************predicted output*************************************"+str(label))
        probabs = loaded_model.predict_proba(test_tfidf)
        confidence_scores={}
        for i, p in enumerate(probabs[0]):
            if (p > 0):
                confidence_scores[labelMap[str(i)]]=p*100
        print("***********************************predicted probabilities*************************************")
        ordered_confidence=sorted(confidence_scores.items(),key=operator.itemgetter(1),reverse=True)
        doc=document.document(labelMap[str(label)],ordered_confidence)
        print("got document :" + doc.predictedlabel)
        return doc
        # except Exception as e:
        #     print('^^^^^^^^^^^^^^^^^^^^^^^^^^Failed to classify using the model^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        #     print(e)

    def getLabelMappings(self):
        key_obj.key = MAPPING_FILE_NAME
        key_obj.get_contents_to_filename(MAPPING_LOCAL_PATH)
        mappingsfile=open(MAPPING_LOCAL_PATH,'r')
        lines=mappingsfile.readlines()
        print(lines)
        for line in lines:
            keyval=line.split(':')
            labelMap[keyval[0]]=keyval[1].strip()
        print('############################Encoded label map for the categories is loaded##########################')
