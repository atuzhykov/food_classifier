from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import pickle
import numpy as np
import pandas as pd


train = pd.read_csv('MFP_scrapped_food.csv')

labels = train.columns.drop(['name', 'class'])
target = train['class']

model = LogisticRegression(penalty='l2',C=0.1)
 
model.fit(train[labels], target)
joblib.dump(model, 'lrmodel.pkl') 




