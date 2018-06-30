# Document-Classification

A classifier to classify the different categories of documents related to mortgage. For eg., if it is a BILL or BINDER or CHANGE OF POLICY etc.This is a REST API which when given the document content as input, asssigns it a category/label along with the confidence scores.<br>
 
Start the flask application by running the app.py and query the endpoints either using cURL or from the browser<br>

API Endpoints<br>
application/json - http://localhost:5000/docjson?words=[document content in text]<br>
text/html - http://localhost:5000/docclassifer and POST the content through the HTML form <br>

An ensemble classification RandomForest is used to train the model. The trained model is saved and loaded in the api.<br>
