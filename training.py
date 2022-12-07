def training_dataset(layer_1, layer_2, layer_3, epoch) :
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

    # Use 70% data for training and 30% for testing
    x_train, x_test, y_train, y_test = train_test_split(data_upsampled[x], 
                                                        data_upsampled[y], 
                                                        test_size=0.3, 
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


    # Sci-kit learn ANN
    # Use 3 hidden layer with 10 unit for each hidden layer
    mlp = MLPClassifier(hidden_layer_sizes=(layer_1,layer_2,layer_3), 
                        activation='relu', 
                        solver='sgd', 
                        max_iter=epoch)

    mlp.fit(x_train,y_train.values.ravel())
    # Save Model
    import joblib

    filename = "model.joblib"
    joblib.dump(mlp, filename)


    training_score = mlp.score(x_train, y_train)

    return training_score    
