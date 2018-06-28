from flask import Flask, render_template, request, jsonify
from action import classifier_action
import logging
app = Flask(__name__)

@app.route('/docclassifer')
def index():
    return render_template('index.html')

@app.route('/docclassifer', methods=['GET'])
def get_class_json():
    text=request.args.get('words')
    print(text)
    doc=actionobj.getlabelfrommodel(text)
    logger.info("Returning category for input document as "+doc.predictedlabel)
    return jsonify(ClassLabel=doc.get_label(),confidence_scores=doc.confidence)

@app.route('/docclassifer', methods=['POST'])
def get_class():
    text=request.form['documenttext']
    doc=actionobj.getlabelfrommodel(text)
    logger.info("Returning category for input document as "+doc.predictedlabel)
    return render_template("index.html",category=doc.predictedlabel, confidence=doc.confidence)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler('DocClassifier.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info('Starting the Flask app on localhost')
    actionobj=classifier_action.classifier_action(logger)
    actionobj.getLabelMappings()
    app.run(debug=False)