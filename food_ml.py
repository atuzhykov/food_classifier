from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.externals import joblib

import pickle
import numpy as np
import pandas as pd

def pipeline(url):
    df = pd.read_csv(url).drop('name', 1)

    df['protein'] = df["protein"].astype(float)
    df["fat"] = df["fat"].astype(float)
    df["carbohydrates"] = df["carbohydrates"].astype(float)
    df["sugar"] = df["sugar"].astype(float)
    df["calories"] = df["calories"].astype(float)

        # data normalization
    # df['protein'] = df["protein"] / df["protein"].max()
    # df["fat"] = df["fat"] / df["fat"].max()
    # df["carbohydrates"] = df["carbohydrates"] / df["carbohydrates"].max()
    # df["sugar"] = df["sugar"] / df["sugar"].max()
    # df["calories"] = df["calories"] / df["calories"].max()

    labels = df.drop('class',1).to_numpy()
    # transformer = RobustScaler().fit(labels)
    # labels = transformer.transform(labels)
    target = df['class']





    models = [  LogisticRegression(penalty='l2',C=0.1), 
                RandomForestClassifier(n_estimators=100),
                GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls'),
                AdaBoostClassifier(n_estimators=100) ]

    for model in models:
        model.fit(labels, target)
        joblib.dump(model, '{}.pkl'.format(type(model).__name__)) 
