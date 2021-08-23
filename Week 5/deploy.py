from flask import Flask, jsonify, request
import pickle
import pandas as pd

app= Flask(__name__)
@app.route('/', methods = ['GET', 'POST']) 
def home():
    if (request.method == 'GET'):
        
        data = "hello world"
        return jsonify({'data': data})
    
@app.route('/predict/')
def price_predict():
    model = pickle.load(open('model.pickle','rb'))
    income = request.args.get('income')
    house_age = request.args.get('house_age')
    rooms = request.args.get('rooms')
    bedrooms = request.args.get('bedrooms')
    population = request.args.get('population')
    
    test_df=pd.DataFrame({'Income':[income], 'House Age':[house_age], 'Rooms':[rooms], 'Bedrooms':[bedrooms], 'Population':[population] })
    
    pred_price = model.predict(test_df)
    return jsonify({'House Price': str(pred_price)})

if __name__ == '__main__':
    app.run(debug = True)