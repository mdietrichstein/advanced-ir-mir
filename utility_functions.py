from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from IPython.core.display import HTML
import itertools
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def show_classification_results(classification_results):
    
    accuracy   = []
    precision  = []
    recall     = []
    f1         = []
    
    for feature_name in classification_results.keys():
        accuracy.append( accuracy_score (classification_results[feature_name][0], classification_results[feature_name][1]))
        precision.append(precision_score(classification_results[feature_name][0], classification_results[feature_name][1], average='macro'))
        recall.append(   recall_score   (classification_results[feature_name][0], classification_results[feature_name][1], average='macro'))
        f1.append(       f1_score       (classification_results[feature_name][0], classification_results[feature_name][1], average='macro'))
        
        
    result = pd.DataFrame({'Featureset': classification_results.keys(),
                           'Accuracy'  : accuracy,
                           'Precision' : precision,
                           'Recall'    : recall,
                           'F1-Score'        : f1})    
    
    result = result[['Featureset', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].sort_values("Accuracy", ascending=False)
    
    return HTML(result.to_html(index=None))


def plot_confusion_matrix(results, encoder):
    
    from sklearn.metrics import confusion_matrix
    
    results = pd.DataFrame({'y_true': encoder.inverse_transform(results[0]),
                            'y_pred': encoder.inverse_transform(results[1])})
    
    results["true_positives"] = (results.y_true == results.y_pred).astype(int)        
    
    cm = confusion_matrix(results.y_true, results.y_pred)

    classes = encoder.classes_

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(6,6))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix', size=16)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    confusions = []

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        confusions.append([classes[i], classes[j], cm[i,j], cm_norm[i,j]])

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def show_query_results(filenames, labels, ranked_index):
    
    from IPython.core.display import display, HTML
    pd.set_option('display.max_colwidth', -1)
    
    output = pd.DataFrame({"filename": filenames, "label": labels})
    output["audio"] = '<audio src="http://127.0.0.1:5555/' + output["label"] + '/' + output["filename"].str.replace("\\","/").str.split("/").apply(lambda x: x[-1]) + '" type="audio/mpeg" controls>'
    output          = output.iloc[ranked_index[:10]]
    output["rank"]  = range(output.shape[0])
    output["rank"]  = output["rank"].astype(str)
    output.set_value(0, "rank", "query")
    output          = output.reset_index()

    return HTML(output[["rank", "index", "filename", "label", "audio"]].to_html(escape=False, index=False))