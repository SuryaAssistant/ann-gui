def training_dataset(hidden_dimension = 20, epoch = 10, train_size = 70) :
    # General library(s)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Machine Learning Library(s)
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Machine Learning Result Metric
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.neural_network import MLPClassifier


    # Load dataset to project
    df = pd.read_csv("./dataset.csv")

    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(df.corr(), annot=True)
    # save image
    plt.savefig('./export/heatmap.png', bbox_inches='tight',)

    # use only 3 parameter ==> Age, Balance, isActiveMember based on Heatmap

    new_category = df[["Age","Balance","IsActiveMember","CreditScore","Exited"]]

    df_new = new_category.copy()

    #Upsampling data
    from sklearn.utils import resample
    exit = df_new[df_new["Exited"] == 1]
    not_exit  = df_new[df_new["Exited"] == 0]

    exit_upsample = resample(exit,
                replace=True,
                n_samples=len(not_exit),
                random_state=99)

    data_upsampled = pd.concat([not_exit, exit_upsample])



    # TRAIN_TEST
    # Create train and test data
    # Determine feature (x) and target (y)

    data_vars = data_upsampled.columns.values.tolist()
    y = ['Exited']
    x = [i for i in data_vars if i not in y]

    tst_size = (100 - train_size)/100
    x_train, x_test, y_train, y_test = train_test_split(data_upsampled[x], 
                                                        data_upsampled[y], 
                                                        test_size=tst_size, 
                                                        random_state=0)

    # Normalize train data
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)

    # Save scaler
    import pickle as pkl
    with open("scaler.pkl", "wb") as outfile:
        pkl.dump(scaler, outfile)

    # During test time
    # Load scaler that was fitted on training data
    with open("scaler.pkl", "rb") as infile:
        scaler = pkl.load(infile)
        x_test = scaler.transform(x_test) 

    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    classifier = Sequential()
    classifier.add(Dense(units=hidden_dimension, kernel_initializer='he_uniform', activation='relu', input_dim = 4))

    ## Menambahkan hidden layer
    classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))

    ## Menambahkan output layer
    classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='relu'))

    ## Compile
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model_history = classifier.fit(x_train, y_train, batch_size=10, validation_split=0.33, epochs=epoch)

    # accuracy history
    plt.figure(figsize=(6,6))
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.savefig('./export/accuracy.png', bbox_inches='tight',)

    ## loss
    plt.figure(figsize=(6,6))
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('./export/loss.png', bbox_inches='tight',)

    # Save Model
    classifier.save('./model/ann.h5')
    final_accuracy = model_history.history["accuracy"][epoch-1]

    #to_predict = [42, 0, 1, 619]
    #to_predict = np.array(to_predict)
    #to_predict = to_predict.reshape(1, -1)
    #to_predict = scaler.transform(to_predict)

    #from tensorflow import keras
    #model = keras.models.load_model('./model/ann.h5')

    #get_result = model.predict(to_predict)
    #print(get_result)

    return final_accuracy


if __name__ == "__main__":
    training_dataset()