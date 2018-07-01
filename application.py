from flask import Flask, render_template, request, jsonify
from action.classifier_action import Action
application = Flask(__name__)
app = application

@app.route('/docclassifier')
def index():
    return render_template('index.html')

@app.route('/docjson', methods=['GET'])
def get_class_json():
    text=request.args.get('words')
    doc=Action.getInstance().getlabelfrommodel(text)
    if(not doc):
        return "Unable to process this document"
    print("##################Returning category for input document as "+doc.get_label()+"####################")
    return jsonify(ClassLabel=doc.get_label(),confidence_scores=doc.get_confidence_scores())

@app.route('/docclassifier', methods=['POST'])
def get_class_html():
    text=request.form['documenttext']
    doc=Action.getInstance().getlabelfrommodel(text)
    if (not doc):
        return "Unable to process this document"
    print("##################Returning category for input document as "+doc.get_label()+"####################")
    return render_template("index.html",category=doc.get_label(), confidence=doc.get_confidence_scores())

print('#############################Starting the Document Classiifer Flask app #################################')
print("***********************************loaded model*************************************")
Action.getInstance().load_model()
Action.getInstance().load_vector()
print("***********************************loaded TFIDF transformer*************************************")
Action.getInstance().getLabelMappings()
# app.run()