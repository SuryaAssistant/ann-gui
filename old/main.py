# Import Libraries
import numpy as np
import pandas as pd

# Dataset
df = pd.read_csv("data.csv")
print(df)

# Analisa Dataset
df.shape
df.columns
df.dtypes
df.head()
df.tail()

# Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Menampilkan jumlah costumer yang keluar dalam data set
plt.figure(figsize=(8,8))
sns.countplot(x="Exited", data=df)
plt.xlabel("0: Costumers masih dengan bank, 1: Costumer keluar dari bank")
plt.ylabel('Count')
plt.title('Bank Costumers Churn Visualizartion')
plt.show()

# Mengecek data yang kosong
df.isna().any()

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df.columns

# Convert setiap fitur ke dalam bentuk angka
geography = pd.get_dummies(df['Geography'], drop_first=True)
gender = pd.get_dummies(df['Gender'], drop_first=True)

# Menambahkan column baru ke dataframe
df = pd.concat([df, geography, gender], axis=1)
df.columns
df.drop(['Geography', 'Gender'], axis=1, inplace=True)

# Data Preprocessing
## Membagi dataset menjadi independent features (x) dan label (y)
x = df.drop('Exited', axis=1)
y = df['Exited']

## Membagi dataset menjadi set train dan test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('x_train size: {}, x_test size: {}'.format(x_train.shape, x_test.shape))

## Aplikasi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Pembuatan ANN
## Import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

## inisiasi ANN
classifier = Sequential()

## Menambahkan layer input dan hidden layer pertama
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim = 11))

## Menambahkan hidden layer kedua
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))

## Menambahkan output layer
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

## Compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## Aplikasi ANN dengan Training set
model_history = classifier.fit(x_train, y_train, batch_size=10, validation_split=0.33, epochs=100)

# accuracy history
plt.figure(figsize=(8,8))
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

## loss
plt.figure(figsize=(8,8))
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Evaluasi Model
## Prediksi Test set
y_pred = classifier.predict(x_test)
print(y_pred)
y_pred = (y_pred > 0.5)
print(y_pred)

# Conffusion Matrux
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#kiri atas = correct prediction customer stay
#kanan bawah = correct prediction customer leave
#kiri bawah = incorrect prediction customer stay
#kanan atas = incorrect prediction customer leave

## Plotting CM
plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='Blues', annot=True, xticklabels=['Tidak Keluar', 'Keluar'], yticklabels=['Tidak Keluar', 'Keluar'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix untuk model ANN')
plt.show()

# Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print('Accuracy model ANN: {}'.format(score))

# Prediksi
def exit_prediction(sample_value):
    sample_value = np.array(sample_value)
    sample_value = sample_value.reshape(1, -1)
    sample_value = sc.transform(sample_value)
    return classifier.predict(sample_value)

# Contoh Prediksi 1
sample_value = [738, 62, 10, 83008.31, 1, 1, 1, 42766.03, 1, 0, 1]
if exit_prediction(sample_value) > 0.5:
    print('Prediksi: Kemungkinan besar keluar!')
else:
    print('Prediksi: Kemungkinan kecil keluar!')