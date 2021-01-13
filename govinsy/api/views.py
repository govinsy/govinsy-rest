from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from datetime import datetime, timedelta 
from time import strftime
import urllib, json, socket, os

# Create your views here.

def index(request):

    # Download and read data
    data = get_data('https://api.covid19api.com/dayone/country/iran', './data/data.json')
    indonesia = get_data('https://api.covid19api.com/dayone/country/indonesia', './data/indonesia.json')

    # Create plot
    create_plot(data)

    # Variable Declaration
    X = data[['Deaths', 'Recovered', 'Active']].values[100:]
    y = data.Confirmed.values[100:]
    data.Date = np.array(data.Date + timedelta(60)).tolist()
    data.Date = list(map(lambda item: str(item.to_pydatetime()), data.Date))
    id_train = indonesia[['Deaths', 'Recovered', 'Active']].values[100:]
    id_test = indonesia.Confirmed.values[100:]
    indonesia.Date = list(map(lambda item: str(item.to_pydatetime()), indonesia.Date))

    # Training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = svm.SVC()
    model.fit(X_train, y_train)
    predictions = model.predict(id_train)
    accuracy = accuracy_score(np.round(predictions/100000), np.round(np.array(id_test)/100000))

    # Comparison plot
    plt.title('Comparison')
    plt.plot(id_test[100:].tolist())
    plt.plot(predictions[100:].tolist())
    plt.legend(['Actual Data', 'Prediction'])
    plt.savefig('./static/img/compare.jpg',  bbox_inches='tight')
    plt.clf()

    # JSON Response
    res = {
        'indonesia': {
            'Confirmed': indonesia.Confirmed.to_list(),
            'Deaths': indonesia.Deaths.to_list(),
            'Recovered': indonesia.Recovered.to_list(),
            'Active': indonesia.Active.to_list(),
            'Date': indonesia.Date.to_list(),
        },
        'prediction': {
            'Confirmed': data.Confirmed[270:].to_list(),
            'Deaths': data.Deaths[270:].to_list(),
            'Recovered': data.Recovered[270:].to_list(),
            'Active': data.Active[270:].to_list(),
            'Date': data.Date[270:].to_list(),
        },
        'predict_plot': request.build_absolute_uri('/static/img/predict.jpg'),
        'compare_plot': request.build_absolute_uri('/static/img/compare.jpg'),
        'accuracy': accuracy
    }
    return HttpResponse(json.dumps(res), content_type="application/json")

def get_data(url, filepath):
    try:
        urllib.request.urlretrieve(url, filepath)
    except:
        print("\nGagal mengambil data terbaru, periksa koneksi!\n") 
    return pd.read_json(filepath)

def create_plot(data):
    date = data.Date + timedelta(60)
    plt.title('Prediction')
    plt.xticks(rotation=90)
    plt.plot(date[270:], data.Confirmed[270:])
    plt.plot(date[270:], data.Deaths[270:])
    plt.plot(date[270:], data.Recovered[270:])
    plt.plot(date[270:], data.Active[270:])
    plt.legend(['Confirmed', 'Deaths', 'Recovered', 'Active'])
    plt.savefig('./static/img/predict.jpg',  bbox_inches='tight')
    plt.clf()