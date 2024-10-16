import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

import numpy as np

diadet_df=pd.read_csv("diabetes_prediction_dataset.csv")

diadet_df.drop_duplicates(inplace=True)

label_encoder=LabelEncoder()
diadet_df["gender"]=label_encoder.fit_transform(diadet_df["gender"])
diadet_df["smoking_history"]=label_encoder.fit_transform(diadet_df["smoking_history"])

X=diadet_df.drop(["diabetes"],axis=1)
Y=diadet_df["diabetes"]


X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.2,random_state=3)

# Пример данных (X - входные данные, y - целевые значения)
X_class = np.array(X_train1)
y_class = np.array(Y_train1)

# Создание модели для классификации
model_class = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(8,)),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Один выход для бинарной классификации
])

model_class.compile(optimizer='adam', loss='binary_crossentropy')

# Обучение модели
model_class.fit(X_class, y_class, epochs=100, batch_size=32)

# Прогноз
#<option value="0" selected>Нет информации</option>
#            <option value="1">Курящий</option>
#            <option value="2">В завязке</option>
#            <option value="3">Бросивший</option>
#            <option value="4">Никогда не курил</option>
#test_data = np.array([[0, 44.0, 0, 0, 4, 19.31, 6.5, 200]])
test_data = np.array([[0, 28.0, 0, 0, 4, 27.32, 5.7, 158]])
y_pred_class = model_class.predict(test_data)
print("Предсказанные значения:", y_pred_class, *np.where(y_pred_class >= 0.5, 'Диабет есть', 'Диабета нет'))
# Сохранение модели для классификации
model_class.save('classification_model.h5')