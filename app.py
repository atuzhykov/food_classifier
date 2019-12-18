from flask_restful import Resource, Api, reqparse
from flask import Flask, jsonify
import datetime
import myfitnesspal
from scipy.special import expit


def food_label_classifier(food_data:dict):
    # very hardcoded model for labeling
    w = 0.005
    x = food_data['calories']
    y = expit(w*x)
 
    result = {
               y < 0.33: 'green',
        0.33 <= y < 0.66: 'yellow',
          0.66 <= y < 1:  'red',
  
    }[True]

    return result
     


def last_day_food_extractor(client):
    day_food_data = [ ]
    day = client.get_date(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day)
    for meal in day.meals:
        for entry in meal:
            food_data = dict()
            food_data['name'] = entry.name
            food_data['calories'] = entry.totals['calories']
            food_data['label'] = food_label_classifier(entry.totals)
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
        result = last_day_food_extractor(client)
        
        return jsonify(result)

api.add_resource(FoodClassifier, '/foodclassifier') 


if __name__ == '__main__':
     app.run()
