import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np

def pipeline(filename):

    df = pd.read_csv('food_dataset.csv').drop(columns=['Food ID', 'Food Name','Sodium','Sugar'])  

    # data preprocessing
    # Each nutrient value in AUSNUT database is presented on a per 100 g edible portion basis.
    df['Protein'] = (df['Protein'].str.split()).apply(lambda x: float(x[0].replace(',', ''))/10)
    df['Fat'] = (df['Fat'].str.split()).apply(lambda x: float(x[0].replace(',', ''))/10)
    df['Carbs'] = (df['Carbs'].str.split()).apply(lambda x: float(x[0].replace(',', ''))/10)
    # The energy value of 1 g of protein is 4 kcal, 1 g of fat is 9 kcal, 1 g of carbohydrates is 3.75 kcal 
    df['Calories'] = df['Protein']*4 + df['Fat']*9 + df['Carbs']*3.75
    X = df.to_numpy()
    kmeans = KMeans(n_clusters=3,n_init=100, max_iter=2000, init='k-means++').fit(X)
    # Output a pickle file for the model
    joblib.dump(kmeans, 'model.pkl') 


