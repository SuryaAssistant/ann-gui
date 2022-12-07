import numpy as np
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

import keras
from keras.models import Sequential
classifier = Sequential()

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