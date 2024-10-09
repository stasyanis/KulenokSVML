
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from neo import SingleNeuron

diadet_df=pd.read_csv("diabetes_prediction_dataset.csv")

diadet_df.drop_duplicates(inplace=True)

label_encoder=LabelEncoder()
diadet_df["gender"]=label_encoder.fit_transform(diadet_df["gender"])
diadet_df["smoking_history"]=label_encoder.fit_transform(diadet_df["smoking_history"])

X=diadet_df.drop(["diabetes"],axis=1)
Y=diadet_df["diabetes"]


X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.2,random_state=3)

# Пример данных (X - входные данные, y - целевые значения)
X = np.array(X_train1)
y = np.array(Y_train1)  # Ожидаемый выход
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=8)
neuron.train(X, y, epochs=5000, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neo_weights.txt')