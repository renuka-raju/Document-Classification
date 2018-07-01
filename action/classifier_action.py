from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.externals import joblib
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from resources import document
import pandas as pd
import operator

class Action:
    __instance = None
    loadedmodel = None
    tfidfvector=None
    labelMap = {}
    BUCKET_NAME = 'mydocclassifierapp'
    VECTOR_FILE_NAME = 'TFIDF.pkl'
    MODEL_FILE_NAME = 'RandomForest.pkl'
    MAPPING_FILE_NAME = 'LabelMapping.txt'
    MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME
    VECTOR_LOCAL_PATH = '/tmp/' + VECTOR_FILE_NAME
    MAPPING_LOCAL_PATH = '/tmp/' + MAPPING_FILE_NAME

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Action.__instance == None:
            __instance=Action()
        return __instance

    @staticmethod
    def load_model():
        try:
            conn = S3Connection()
            bucket = conn.get_bucket(Action.BUCKET_NAME)
            key_obj = Key(bucket)
            key_obj.key = Action.MODEL_FILE_NAME
            key_obj.get_contents_to_filename(Action.MODEL_LOCAL_PATH)
            Action.loadedmodel= joblib.load(Action.MODEL_LOCAL_PATH)
        except Exception as e:
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Problem loading trained model file^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print(e)

    @staticmethod
    def load_vector():
        try:
            conn = S3Connection()
            bucket = conn.get_bucket(Action.BUCKET_NAME)
            key_obj = Key(bucket)
            key_obj.key = Action.VECTOR_FILE_NAME
            key_obj.get_contents_to_filename(Action.VECTOR_LOCAL_PATH)
            Action.tfidfvector= joblib.load(Action.VECTOR_LOCAL_PATH)
        except Exception as e:
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Problem loading TFIDF vector(dimensions from training)^^^^^^^^^^^^^^^^^^^^^^^")
            print(e)

    def getlabelfrommodel(Action,text):
        try:
            tfidf_transformer = TfidfTransformer()
            loaded_vec = CountVectorizer(vocabulary=Action.tfidfvector)
            test = pd.DataFrame()
            test['doc'] = [text]
            test_tfidf = tfidf_transformer.fit_transform(loaded_vec.transform(test['doc']))
            print("***********************************transformed test input*************************************")
            label = Action.loadedmodel.predict(test_tfidf)[0]
            print("***********************************predicted output*************************************"+str(label))
            print(Action.labelMap)
            probabs = Action.loadedmodel.predict_proba(test_tfidf)
            confidence_scores={}
            for i, p in enumerate(probabs[0]):
                if (p > 0):
                    confidence_scores[Action.labelMap[str(i)]]=p*100
            print("***********************************predicted probabilities*************************************")
            ordered_confidence=sorted(confidence_scores.items(),key=operator.itemgetter(1),reverse=True)
            doc=document.document(Action.labelMap[str(label)],ordered_confidence)
            print("Predicted category for the input document as:" + doc.predictedlabel)
            return doc
        except Exception as e:
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^Failed to classify using the model^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            print(e)

    @staticmethod
    def getLabelMappings():
        try:
            conn = S3Connection()
            bucket = conn.get_bucket(Action.BUCKET_NAME)
            key_obj = Key(bucket)
            key_obj.key = Action.MAPPING_FILE_NAME
            key_obj.get_contents_to_filename(Action.MAPPING_LOCAL_PATH)
            mappingsfile=open(Action.MAPPING_LOCAL_PATH,'r')
            lines=mappingsfile.readlines()
            # print(lines)
            for line in lines:
                keyval=line.split(':')
                Action.labelMap[keyval[0]]=keyval[1].strip()
            print('############################Encoded label map for the categories is loaded##########################')
        except Exception as e:
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Problem loading label encodings map file^^^^^^^^^^^^^^^^^^^^^^^")
            print(e)