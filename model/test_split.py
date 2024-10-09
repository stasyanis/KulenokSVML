import os
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.realpath(__file__))


def get_gender_set():
    dataset_df = pd.read_csv(rf'{current_dir}\hight.csv')
    X = dataset_df.drop(['Gender'], axis=1)
    y = dataset_df['Gender']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return y_test, X_test


def get_shoe_size_set():
    dataset_df = pd.read_excel(rf'{current_dir}\legs.xlsx')
    X = dataset_df.drop(['Размер ноги (EU)'], axis=1)
    y = dataset_df['Размер ноги (EU)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return y_test, X_test


def get_heights_test_set():
    dataset_df = pd.read_csv(rf'{current_dir}\data_belita.csv')
    X = dataset_df.drop(['Status Gizi'], axis=1)
    y = dataset_df['Status Gizi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return y_test, X_test
