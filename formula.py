def food_index(food_data):
    protein = food_data['protein']
    fats = food_data['fat']
    carbs = food_data['carbohydrates']
    sugar = food_data['sugar']
    calories = food_data['calories']
    sodium = food_data['sodium']
    index = abs(1 - (carbs/(fats+protein))) + abs(calories*0.1 - sugar) 
    return index
 


  

