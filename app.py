from flask_restful import Resource, Api, reqparse
from flask import Flask, jsonify
import datetime
import myfitnesspal
from sklearn.externals import joblib
import collections
import numpy as np
import math
from food_ml import pipeline
import os
from sklearn.impute import KNNImputer
import pandas as pd

# Only for HEROKU deployment due to daily erasing all filesystem
url = 'https://raw.githubusercontent.com/atuzhykov/food_classifier/master/MFP_scrapped_food_without_names.csv'

import os.path
if not os.path.isfile('GradientBoostingRegressor.pkl'):
    pipeline(url)




def KNN_imputer(food_data,missed_features):
    features = ['protein','fat','carbohydrates','sugar','sodium','calories']
  
   
    Y = pd.read_csv(url).drop('class',1).to_numpy()
    nan = np.nan
    protein = nan if 'protein' not in food_data else food_data['protein']
    fat =  nan if 'fat' not in food_data else food_data['fat']
    carbohydrates =  nan if 'carbohydrates' not in food_data else food_data['carbohydrates']
    sugar = nan if 'sugar' not in food_data else  food_data['sugar']
    sodium =  nan if 'sodium' not in food_data else food_data['sodium']
    calories =  nan if 'calories' not in food_data else food_data['calories']
    print('Vector before restoring {}'.format(np.array([[protein,fat,carbohydrates,sugar,sodium,calories]])))
            

    Y = np.concatenate((Y, np.array([[protein,fat,carbohydrates,sugar,sodium,calories]])))
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X = imputer.fit_transform(Y)[-1].reshape(1, -1)
    print('Restored via KNNImputer vector {}'.format(X))
    return X


def food_label_classifier(X, algo = 'rf'):

    labels = ["green","yellow","red"]

    if algo == 'formula':
        w =[0.4, 0.2, 0.1 ]
        index =  w[0]*X[0,2] + w[1]*X[0,1] + w[2]*X[0,0] + abs(X[0,5]*0.1 - X[0,3]) + math.exp(0.0001*X[0,4])
    
        label = {
               index < 9: 'green',
            9 <= index < 25:   'yellow',
            25 <= index:  'red',
     
        }[True]

        return label

    if algo == 'lr':
        classifier = joblib.load('LogisticRegression.pkl')
        return labels[classifier.predict(X).tolist()[0]]
    
    elif algo == 'gb':
        classifier = joblib.load('GradientBoostingRegressor.pkl')
        return labels[round(classifier.predict(X).tolist()[0])]

    elif algo == 'rf':
        classifier = joblib.load('RandomForestClassifier.pkl')
        return labels[classifier.predict(X).tolist()[0]]

    elif algo == 'ab':
        classifier = joblib.load('AdaBoostClassifier.pkl')
        return labels[classifier.predict(X).tolist()[0]]

def meta_classifier(food_data):
    food_name = food_data.name
    food_data = food_data.totals
    features = ['protein','fat','carbohydrates','sugar','sodium','calories']
    missed_features = []
    for feature in features:
        if feature not in food_data:
            missed_features.append(feature)
    
    if missed_features:
        print('following features are missing for {}: {}'.format(food_name,' '.join(missed_features)))
        X = KNN_imputer(food_data, missed_features).reshape(1, -1)

    else:
        X = np.array([food_data['protein'],food_data['fat'],food_data['carbohydrates'],food_data['sugar'],food_data['sodium'],food_data['calories']]).reshape(1, -1)
    
    algos = ['rf','ab','lr','gb','formula']
    decisions = []
    for algo in algos:
        decisions.append(food_label_classifier(X,algo=algo))
    # return tuple of vector and prediction label
    return (X,collections.Counter(decisions).most_common(1)[0][0])
    

 

     


def food_extractor(client, date):
    day_food_data = [ ]
    generalized_meal = np.zeros(6)
    day = client.get_date(date.year, date.month, date.day)
    for meal in day.meals:
        food_type = meal.name 
        for entry in meal:
            food_data = dict()
            food_data['name'] = entry.name
            food_data['type'] = food_type
            food_data['calories'] = entry.totals['calories']
            food_data['label'] = meta_classifier(entry)[1]
            food_data['day'] = date.strftime("%d/%m/%Y")
            day_food_data.append(food_data)
            
            # getting vector (thats nedeed in case it was restored)
            X = meta_classifier(entry)[0]
            for idx, j in np.ndenumerate(generalized_meal):
                generalized_meal[idx] += X[0][idx]
 
           
            

    if day_food_data:
        generalized_meal = np.divide(generalized_meal,len(food_data)).reshape(1, -1)
        algos = ['rf','ab','lr','gb','formula']
        decisions = []
        for algo in algos:
            decisions.append(food_label_classifier(X,algo=algo))
        day_food_data.append({'daylabel': collections.Counter(decisions).most_common(1)[0][0]})

    
    return day_food_data



app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('email', type=str)
parser.add_argument('password', type=str)
parser.add_argument('date', type=str)


class FoodClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        email = args['email'].strip()
        password = args['password'].strip()
        date = datetime.datetime.strptime(args['date'], '%Y-%m-%d')
        client = myfitnesspal.Client(username=email, password=password)
        print('logged as {}'.format(email))
        result = food_extractor(client,date)
        
        return jsonify(result)

api.add_resource(FoodClassifier, '/foodclassifier') 




if __name__ == '__main__':
    app.run()
