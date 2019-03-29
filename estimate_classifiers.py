from cnn_classifier import Load_Data
from cnn_classifier import CNN_Classify
from rnn_classifier import RNN_Classify

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

## Loading data
x_train, x_test, y_train, y_test, classes = Load_Data()

## CNN classification
print("")
print("CNN Started...")
y_pred_CNN, y_pred_CNN_label = CNN_Classify(x_train, y_train, x_test, y_test, 0, classes)

print("testing data printed")
print(x_test[1])

## RNN classification
print("")
print("RNN Started...")
y_pred_RNN, y_pred_RNN_label = RNN_Classify(x_train, y_train, x_test, y_test, 0, classes)

## Print results
target_names = [str(classes[1]), str(classes[0])]
print("")
print("CNN Result...")
print(classification_report(np.argmax(y_test, axis=1), y_pred_CNN, target_names=target_names, digits=4))
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred_CNN))

print("")
print("RNN Result...")
print(classification_report(np.argmax(y_test, axis=1), y_pred_RNN, target_names=target_names, digits=4))
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred_RNN))

## Writting csv file
import csv
with open("result.csv", "w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["Original Label", "CNN Prediction", "RNN Prediction"])
    for i in range(y_test.shape[0]):
        row = [str(classes[1 - int(y_test[i][1])]), str(y_pred_CNN_label[i]), str(y_pred_RNN_label[i])]
        writer.writerow(row)
print("Writing completed...Please check result.csv...")
