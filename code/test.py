import pycrfsuite
from sklearn.model_selection import train_test_split
import json
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

from phrase_based_model import test_set, predict, tag_mapping

# test data set
test_tokens = [tokens for tokens, _ in test_set]
test_labels = [labels for _, labels in test_set]

true_labels = []
predicted_labels = []

for tokens, true_tags in test_set:
    pred = predict(tokens)  # returns list of (token, predicted_tag)
    pred_tags = [tag for _, tag in pred]  # extract only tags
    predicted_labels.extend(pred_tags)
    true_labels.extend([tag_mapping[tag] for tag in true_tags])  # convert true label indices to tag strings

# model evaluation
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2%}")

# confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=list(tag_mapping.values()))
print("Confusion Matrix:")
print(cm)

# classification report
report = classification_report(true_labels, predicted_labels)
print("Classification Report:")
print(report)
