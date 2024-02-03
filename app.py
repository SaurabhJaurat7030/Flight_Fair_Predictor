from flask import Flask,render_template, request
import pickle, pandas as pd, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/', methods=["GET", "POST"])
def Home():
    price = 0
    # features = {}
    if request.method == "POST":

        features = {
            "airline" : request.form["airline"],
            "source" : request.form["source"],
            "destination" : request.form["destination"],
            "total_stops" : request.form["stops"],
            "dep_timedate" : request.form["deptime"],
            "arrival_timedate" : request.form["arrtime"]
        }

        airline =  features['airline']
        source= features['source']
        destination=  features['destination']
        total_stops=  features['total_stops']
        dep_timedate =  features['dep_timedate']
        arrival_timedate=  features['arrival_timedate']

        # ------------------- Categorical Variables ---------------------
        # Airline
        airline_list = ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business', 
                'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 
                'Vistara', 'Vistara Premium economy']
        airline_arr = [0] * len(airline_list)
        airline_enm = enumerate(airline_list)
        for index, value in airline_enm:
            if value == airline:
                airline_arr[index] = 1 
        airline_arr = np.array(airline_arr)


        # Source
        source_list = ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']
        source_arr = [0] * len(source_list)
        source_enm = enumerate(source_list)
        for index, value in source_enm:
            if value == source:
                source_arr[index] = 1
        source_arr = np.array(source_arr)


        # Destination
        dest_list= ['Cochin', 'Delhi', 'Hyderabad','Kolkata', 'New Delhi']
        dest_arr = [0] * len(dest_list)
        dest_enm = enumerate(dest_list)
        for index, value in dest_enm:
            if value == destination:
                dest_arr[index] = 1
        destination_arr = np.array(dest_arr)

        # ======================= creating array of the numarical varible ===========================
        date_format = "%Y-%m-%dT%H:%M"


        dep_datetime_object = datetime.strptime(dep_timedate, date_format)
        # dep_year = dep_datetime_object.year
        dep_month = dep_datetime_object.month
        dep_day = dep_datetime_object.day
        dep_hour = dep_datetime_object.hour
        dep_minutes = dep_datetime_object.minute

        arr_datetime_object = datetime.strptime(arrival_timedate, date_format)
        # arr_year =arr_datetime_object.year
        # arr_month =arr_datetime_object.month
        # arr_day = arr_datetime_object.day
        arr_hour = arr_datetime_object.hour
        arr_minutes = arr_datetime_object.minute

        duration = abs((dep_datetime_object - arr_datetime_object).total_seconds())/60

        # creating array
        num_features = np.array([total_stops, dep_month, dep_day, dep_hour, dep_minutes, arr_hour, arr_minutes, duration])

        # concatenating all array to make input array
        input_array = np.concatenate((num_features, airline_arr,source_arr,destination_arr)).astype(float)
      

        # importing model pickle file
        with open('flight_rf.pkl', 'rb') as file:
            predictor = pickle.load(file)

        # prediction
        input_array = input_array.reshape(1,-1)
        price = round(predictor.predict(input_array)[0])
        
        print(price)
    return render_template("index.html", price=price)



if __name__ == '__main__':
    app.run(debug=True)