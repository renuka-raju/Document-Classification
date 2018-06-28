class document:
    def __init__(self, label, confidence):
        self.predictedlabel = label
        self.confidence=confidence

    def get_label(self):
        return self.predictedlabel

    def get_confidence_scores(self):
        return self.confidence
