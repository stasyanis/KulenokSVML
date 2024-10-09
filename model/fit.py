import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

tweet_df = pd.read_csv("tweet_emotions.csv")
label_encoder=LabelEncoder()
tweet_df["content"]=label_encoder.fit_transform(tweet_df["content"])
tweet_df["sentiment"]=label_encoder.fit_transform(tweet_df["sentiment"])
X = tweet_df["content"]
Y = tweet_df["sentiment"]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.3, random_state=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train1, Y_train1)
with open('tweet_pickle', 'wb') as pkl:
    pickle.dump(model, pkl)
