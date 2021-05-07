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
datelist=dataset["Date"].tolist()
openlist=dataset["Open"].tolist()
highlist=dataset["High"].tolist()
lowlist=dataset["Low"].tolist()
adlist=dataset["Adj close"].tolist()
vollist=dataset["Volume"].tolist()
leng=len(datelist)
open2list=[["Date","Open"]]
high2list=[["Date","High"]]
low2list=[["Date","Low"]]
ad2list=[["Date","Adjacent"]]
vol2list=[["Date","Volume"]]
for i in range(0,len(datelist)):
    open2list.append([datelist[i],openlist[i]])
    high2list.append([datelist[i],highlist[i]])
    low2list.append([datelist[i],lowlist[i]])
    ad2list.append([datelist[i],adlist[i]])
    vol2list.append([datelist[i],vollist[i]])
from sklearn.preprocessing import MinMaxScaler
scalerx = MinMaxScaler(feature_range=(0,1))
scalery = MinMaxScaler(feature_range=(0,1))
scaled_data = dataset.iloc[:, [1, 2, 3, 4, 5, 6]]
scaled_data=pd.DataFrame(scaled_data,columns=["Open","High","Low","Close","Adj close","Volume"])
X=scaled_data.iloc[:, [0, 1, 2, 4, 5]]
y=scaled_data.iloc[:,[3]]
y=scalery.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
X_train, y_train, y_test=np.array(X_train),np.array(y_train),np.array(y_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
model=Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)
'''predictions=model.predict(X_test)
predictions=pd.DataFrame(predictions)
y_test=pd.DataFrame(y_test)
y_test=scalery.inverse_transform(y_test)
#y_test=y_test.to_numpy()
predictions=scalery.inverse_transform(predictions)
#predictions=predictions.to_numpy()
############'''
def homePageView(request):
    return HttpResponse('Hello, World!')

def formss(request):
    global X_test
    if(request.method=='POST'):
        student=StudentForm(request.POST)
        if(student.is_valid()):
            open=student.cleaned_data.get("open")
            high=student.cleaned_data.get("high")
            low=student.cleaned_data.get("low")
            adjacent=student.cleaned_data.get("adjacent")
            vol=student.cleaned_data.get("vol")
            X_test.loc[len(X_test)]=[open,high,low,adjacent,vol]
            X_test=scalerx.fit_transform(X_test)
            X_test=np.array(X_test)
            X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            ans=model.predict(X_test)
            ans=scalery.inverse_transform(ans)
            ans[0][0]=ans[len(ans)-1][0]
            print(open,high,low,adjacent,vol,ans)
    else:
        student = StudentForm()
        ans=[[0]]
    passd={'form':student,
    'ans':ans,
    'date':datelist,'open':open2list,'high':high2list,'low':low2list,'adj':ad2list,'vol':vol2list}
    return render(request,"Stock/index.html/",passd)
def hello(request):
    return render(request,"Stock/index.html/")
def logo(request):
    return render(request,"Stock/logo.png/")
def js(request):
    return render(request,"Stock/cht.js/")
