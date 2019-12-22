from flask_restful import Resource, Api, reqparse
from flask import Flask, jsonify
import datetime
import myfitnesspal
from sklearn.externals import joblib
import collections
import numpy as np
import math



# Only for HEROKU deployment due to daily erasing all filesystem
# import os.path
# if not os.path.isfile('model.pkl'):
#     url='https://raw.githubusercontent.com/atuzhykov/food_classifier/master/food_dataset.csv'
#     pipeline(url)

def food_index(food_data):
    protein = food_data['protein']
    fats = food_data['fat']
    carbs = food_data['carbohydrates']
    sugar = food_data['sugar']
    calories = food_data['calories']
    sodium = food_data['sodium']
    w =[0.4, 0.2, 0.1 ]
    index =  w[0]*carbs + w[1]*fats + w[2]*protein + abs(calories*0.1 - sugar) + math.exp(0.0001*sodium)
    
    label = {
               index < 9: 'green',
        9 <= index < 25:   'yellow',
          25 <= index:  'red',
     
    }[True]

    return label

def food_label_classifier(food_data:dict, algo = 'rf'):
    labels = ["green","yellow","red"]
    X = np.array([food_data['protein'],food_data['fat'],food_data['carbohydrates'],food_data['sugar'],food_data['calories']]).reshape(1, -1)

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
    algos = ['rf','ab','lr','gb']
    decisions = []
    for algo in algos:
        decisions.append(food_label_classifier(food_data,algo=algo))
    decisions.append(food_index(food_data))
    return collections.Counter(decisions).most_common(1)[0][0]
    

 

     


def last_day_food_extractor_formula(client):
    day_food_data = [ ]
    day = client.get_date(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day)
    for meal in day.meals:
        for entry in meal:
            food_data = dict()
            food_data['name'] = entry.name
            food_data['calories'] = entry.totals['calories']
            food_data['label'] = meta_classifier(entry.totals)
            day_food_data.append(food_data)

    return day_food_data



app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('email', type=str)
parser.add_argument('password', type=str)


class FoodClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        email = args['email']
        password = args['password']
        client = myfitnesspal.Client(username=email, password=password)
        result = last_day_food_extractor_formula(client)
        
        return jsonify(result)

api.add_resource(FoodClassifier, '/foodclassifier') 



if __name__ == '__main__':
    app.run()
