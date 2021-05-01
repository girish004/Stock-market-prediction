from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.forms import *
from Stock.forms import *
# Create your views here.
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
import bs4
from fastnumbers import isfloat
from fastnumbers import fast_float
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
import seaborn as sns
from requests.exceptions import HTTPError
import json
from bs4 import BeautifulSoup
dataset = pd.read_excel ('D:\Girish\AI\Project\stock_data.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
dataset=dataset[["Date","Open","High","Low","Close","Adj close","Volume"]]
plt.figure(figsize=(6,4))
plt.title("Open and close prize comparison for the entire data")
plt.plot(dataset["Close"],label="Close")
plt.plot(dataset["Open"],label="Open")
plt.show()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset.iloc[:, [1, 2, 3, 4, 5, 6]])
scaled_data=pd.DataFrame(scaled_data,columns=["Open","High","Low","Close","Adj close","Volume"])
testing_set=scaled_data.iloc[1500:, [0, 1, 2, 3, 4, 5]]
X=scaled_data.iloc[:, [0, 1, 2, 4, 5]]
y=scaled_data.iloc[:,3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
X_train, X_test, y_train, y_test=np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
testing_set=np.array(testing_set)
testing_set=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))
model=Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
sc = StandardScaler()
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)
############
def homePageView(request):
    return HttpResponse('Hello, World!')

def formss(request):
    student = StudentForm()
    name=""
    return render(request,"Stock/index.html/",{'form':student})
def hello(request):
    return render(request,"Stock/index.html/")
