import numpy as np
import matplotlib.pyplot as plt
import itertools

import winsound
from datetime import datetime

from sklearn import metrics
from sklearn.metrics import \
precision_recall_curve,\
f1_score,\
auc, \
accuracy_score, \
classification_report, \
confusion_matrix, \
roc_curve, \
roc_auc_score,\
recall_score,\
make_scorer,\
ConfusionMatrixDisplay


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def score_model(y_test, y_pred):
    """Prints recall scores and confusion matrix.
    Returns recall scores.
    """
    ac = accuracy_score(y_test, y_pred)
    print('Accuracy: ', ac, '\n')

    f1 = f1_score(y_test, y_pred, average=None)
    print('f1-score: ', f1, '\n')

    rec_scores = recall_score(y_test, y_pred, average=None)
    print('Recall-scores: ', rec_scores, '\n')


    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ['No diabetes','Prediabetes','Diabetes'])

    return recall_score(y_test, y_pred, average=None)

def predict_and_score(model):
    """
    Predicts and prints metrics for a model.  Uses global train/test sets.
    """
    global y_tr_pred, y_te_pred

    y_tr_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    score_model(y_test, y_pred)

def fit_predict_score(model, model_type, params, beep=True):
    """
    Fits, predicts, and prints metrics for a model.  Uses global train/test sets.
    Returns fitted_model.
    """
    global y_tr_pred, y_pred

    start = datetime.now()

    model = model_type(**params)
    fitted_model = model.fit(X_train, y_train)

    predict_and_score(model)

    stop = datetime.now()

    if beep:
        duration = 300  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)

    print('\nStart time: ', start)
    print('\nStop time: ', stop)
    print('\nRuntime: ', stop-start)

    return fitted_model

def start_timer():
    global start_time
    start_time = datetime.now()
    print('Start time: ', start_time.strftime('%H:%M:%S'))

def end_timer():
    end_time = datetime.now()
    print('End time: ', end_time.strftime('%H:%M:%S'))
    runtime = end_time - start_time
    print('Runtime: ', runtime,'\n')

def beep_when_done():
    duration = 300  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
