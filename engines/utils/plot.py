import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, num_classes, label_list, title='Confusion Matrix'):
    classes = [str(i) for i in range(num_classes)]
    labels = range(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, label_list, fontsize=20)
    plt.yticks(tick_marks, label_list, fontsize=20)
    print("Confusion Matrix")
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label', fontsize=30)
    plt.xlabel('Predicted Label', fontsize=30)
    plt.show()
