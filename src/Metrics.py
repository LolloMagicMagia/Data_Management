class Metrics:
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    auc = 0

    def __init__(self, accuracy, precision, recall, f1_score, auc):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.auc = auc