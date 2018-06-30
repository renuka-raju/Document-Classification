data = open('shuffled-full-set-hashed.csv').read()
labels, docs = [], []
count=0
for i, line in enumerate(data.split("\n")):
    content = line.split(",")
    count+=1;
    if len(content)==2 and content[0].isupper():
        labels.append(content[0])
        docs.append(content[1])

#***********************checking imbalance in class labels*************************
# labelsorted=sorted(labels)
# from itertools import groupby
# X=[len(list(group)) for key, group in groupby(labelsorted)]
# X=sorted(X,reverse=True)
# print(X)
# print(max(X),len(X),len(labels))

import pandas as pd
import numpy
training=pd.DataFrame()
training['doc']=docs
training['label']=labels
# print(training['doc'])
# print(training['label'])
from sklearn import model_selection, preprocessing, naive_bayes, metrics, ensemble, decomposition
#
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
train_features, valid_features, train_label, valid_label = model_selection.train_test_split(training['doc'], training['label'],test_size=0.10)
#
# print(len(train_features),len(train_label))
count_vect = CountVectorizer(analyzer='word',min_df=0.02, max_df=0.95)
# tfidf_vect = TfidfVectorizer(analyzer='word',min_df=0.01, max_df=0.9, smooth_idf=True)
tfidf_trans = TfidfTransformer()
# tfidf_vect.fit(train_features)
count_vect.fit(training['doc'])
train_tfidf = tfidf_trans.fit_transform(count_vect.transform(train_features))
# print(train_tfidf.shape)
valid_tfidf = tfidf_trans.transform(count_vect.transform(valid_features))
print(valid_tfidf.shape)
# tfidf=tfidf_vect.get_feature_names()

encoder = preprocessing.LabelEncoder()
train_label = encoder.fit_transform(train_label)
valid_label = encoder.fit_transform(valid_label)
decoded=encoder.inverse_transform(train_label)
dictmap={}
labelmapping=open('LabelMapping.txt','w')
for x,y in zip(list(train_label),list(decoded)):
    dictmap[x]=y
for key,val in dictmap.items():
    labelmapping.write(str(key)+":"+str(val))
    labelmapping.write('\n')
labelmapping.close()
# print(train_label.shape)
# print(valid_label.shape)
# #**********************Naive Bayes text classification***************************************
# nbclassifier=naive_bayes.MultinomialNB()
# nbclassifier.fit(train_tfidf, train_label)
# predictions = nbclassifier.predict(valid_tfidf)
# print(nbclassifier.predict_proba(valid_tfidf))
#**************************Bagging Model-Random Forest*************************************
bagclassifier=ensemble.RandomForestClassifier()
bagclassifier.fit(train_tfidf, train_label)
predictions = bagclassifier.predict(valid_tfidf)
print(predictions[1:10])
print(valid_label[1:10])
print(metrics.accuracy_score(predictions, valid_label))
print(metrics.classification_report(valid_label, predictions))
print(metrics.confusion_matrix(valid_label, predictions))
# print(bagclassifier.predict_proba(valid_tfidf))
###############################################################################################
import pickle
model=open('RandomForest.pkl','wb')
pickle.dump(bagclassifier,model)
model.close()
tfidf=open('TFIDF.pkl','wb')
pickle.dump(count_vect.vocabulary_,tfidf)
tfidf.close()