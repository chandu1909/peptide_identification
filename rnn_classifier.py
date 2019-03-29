from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow

def Load_Data():
    ## Load csv file
    print('Loading data...')
    Tk().withdraw()
    file_name = askopenfilename(filetypes = (("CSV File", "*.csv"), ("All Files", "*.*")),
                                title = "Choose a dataset csv file.")
    dataset = pd.read_csv(file_name, header = None).values
    x_data = dataset[1 :, : -1]
    y_temp_data = dataset[1 :, -1]

    ## Preprocessing
    print("Preprocessing...")

    # The dataset includes string variables and they need to be converted
    # into numerical values.For this, we used LabelEncoder of sklearn preprocessing.By
    # using this function, non - numerical fields are converted into floating numbers.
    enc = preprocessing.LabelEncoder()
    for i in range(x_data.shape[1]):
        # if not(str(x_data[0, i]).replace('.', '').isnumeric()):
            enc.fit(x_data[:, i])
            x_data[:, i] = enc.transform(x_data[:, i])

    x_data = np.array(x_data, dtype = float)
    y_temp_data = np.array(y_temp_data, dtype = float)
    y_data = np.ones((y_temp_data.shape[0], 2))
##    y_data[:, 0] = y_temp_data
##    y_data[:, 1] = y_data[:, 1] - y_temp_data
    y_data[:, 0] = np.array(y_temp_data == 1, dtype = float)
    y_data[:, 1] = np.array(y_temp_data != 1, dtype = float)
    classes = np.unique(y_temp_data)

    ## Normalization
    print("Normalizing...")

    # Data normalization is essential for good performance of classifier.We transform
    # features by scaling each feature to a given range.
    # This estimator scales and translates each feature individually such that
    # it is in the given range on the training set, e.g.between zero and one.
    min_max_scaler = preprocessing.MinMaxScaler()
    x_data = min_max_scaler.fit_transform(x_data)

    ## Split train and test data
    #  Ratio on train and test
    print("Collecting train and test data...")

    # The dataset is split into train and test data with ratio of 7: 3.
    # Then each dataset is reshaped to be suitable for classifier.
    ratio = 0.7
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 1 - ratio, random_state = 42)

    print('x_train shape: {}'.format(x_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('x_test shape: {}'.format(x_test.shape))
    print('y_test shape: {}'.format(y_test.shape))

    ## Reshape data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    print('train shape after reshape: {}'.format(x_train.shape))
    print('test shape after reshape: {}'.format(x_test.shape))
    return x_train, x_test, y_train, y_test, classes

def RNN_Classify(x_train, y_train, x_test, y_test, verbose, classes):
    ## Size of parameters
    batch_size = 10
    num_classes = len(np.unique(y_train))
    epochs = 10
    droprate = 0.50

    ## Start Recurrent Neural Network
    # Fully connected RNN
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.SimpleRNN(units = 128, input_shape = (x_train.shape[1], x_train.shape[2]), return_sequences=False))


    # Final layer
    model.add(tensorflow.keras.layers.Dense(2))
    model.add(tensorflow.keras.layers.Activation('sigmoid'))

    # In training, we used RMSprop method and binary cross entropy criteria to measure loss.
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    ## Train and test
    model_path='rnn_model.h5'

    # prepare callbacks
    callbacks = [
        tensorflow.keras.callbacks.EarlyStopping(
            monitor='val_acc',
            patience=10,
            mode='max',
            verbose=1),
        tensorflow.keras.callbacks.ModelCheckpoint(model_path,
            monitor='val_acc',
            save_best_only=True,
            mode='max',
            verbose=0)
    ]

    history = model.fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epochs,
              verbose = verbose,
              validation_data=(x_train, y_train), shuffle=True, callbacks=callbacks)

    score = model.evaluate(x_test, y_test, verbose=0)

    # print loss and accuracy
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    ## Print result
    y_pred = model.predict(x_test)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)

    y_pred = model.predict_classes(x_test)
    y_pred_label = y_pred.copy()
    for i in range(len(y_pred)):
        y_pred_label[i] = classes[1 - y_pred_label[i]]
    print(y_pred_label)

    p=model.predict_proba(x_test)
    return y_pred, y_pred_label

def main():
    print('RNN Classification started...')

    ## Loading data
    x_train, x_test, y_train, y_test, classes = Load_Data()

    ## RNN classification
    y_pred, y_pred_label = RNN_Classify(x_train, y_train, x_test, y_test, 1, classes)

    ## Print results
    target_names = [str(classes[1]), str(classes[0])]
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names, digits=4))
    print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

    ## Writting csv file
    import csv
    with open("rnn result.csv", "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Original Label", "RNN Prediction"])
        for i in range(y_test.shape[0]):
            row = [str(classes[1 - int(y_test[i][1])]), str(y_pred_label[i])]
            writer.writerow(row)
    print("Writing completed...Please check rnn result.csv...")

if __name__ == "__main__":
    main()
