from flask import Flask, render_template, request, jsonify
from action import classifier_action
application = Flask(__name__)
app = application

actionobj=classifier_action.classifier_action()

@app.route('/docclassifier')
def index():
    return render_template('index.html')

@app.route('/docjson', methods=['GET'])
def get_class_json():
    text=request.args.get('words')
    doc=actionobj.getlabelfrommodel(text)
    if(not doc):
        return "Unable to process this document"
    print("##################Returning category for input document as "+doc.get_label()+"####################")
    return jsonify(ClassLabel=doc.get_label(),confidence_scores=doc.get_confidence_scores())

@app.route('/docclassifier', methods=['POST'])
def get_class_html():
    text=request.form['documenttext']
    doc=actionobj.getlabelfrommodel(text)
    if (not doc):
        return "Unable to process this document"
    print("##################Returning category for input document as "+doc.get_label()+"####################")
    return render_template("index.html",category=doc.get_label(), confidence=doc.get_confidence_scores())

if __name__ == '__main__':
    print('#############################Starting the Flask app on localhost#################################')
    actionobj.getLabelMappings()
    app.run(debug=False)