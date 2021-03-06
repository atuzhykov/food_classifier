
import myfitnesspal
import pandas as pd
import time


email = 'anarchypunk@ukr.net'
password = 't*u4C8iWAB5Trn9'
client = myfitnesspal.Client(username=email, password=password)

with open('requests.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

dataset = []
for item in content:
    nitem  = list(map(str, item.replace('(','').replace(')','').split(',')))
    food_items = client.get_food_search_results(nitem[0])
    time.sleep(150)
    print(nitem[0])

    for mfpids in food_items:
        food_data = dict()
        meal = client.get_food_item_details(mfpids.mfp_id)
        food_data['name'] = meal.name
        food_data['protein'] = meal.protein
        food_data['fat'] = meal.fat
        food_data['carbohydrates']= meal.carbohydrates
        food_data['sugar']= meal.sugar
        food_data['sodium']= meal.sodium
        food_data['calories'] = meal.calories
        food_data['class'] = nitem[1]
        dataset.append(food_data)
        df = pd.DataFrame(dataset)
        df.to_csv('MFP_scrapped_food.csv')


