import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np

def pipeline(filename):

    df = pd.read_csv(filename).drop(columns=['name'])
    df = df.drop(df.columns[0], axis=1)  
    # data preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(df.to_numpy())
    print(X)
    kmeans = KMeans(n_clusters=3,n_init=100, max_iter=2000, init='k-means++').fit(X)
    # Output a pickle file for the model
    joblib.dump(kmeans, 'model.pkl') 


pipeline('MFP_scrapped_food.csv')

  


