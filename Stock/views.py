from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.forms import *
from Stock.forms import *
# Create your views here.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
from html.parser import HTMLParser
from selenium import webdriver

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from functools import reduce
from sklearn.metrics import mean_absolute_error
def get_table_simple(table,is_table_tag=True):
    elems = table.find_all('tr') if is_table_tag else get_children(table)
    row_elems = table.find_all("td")
    i=0
    table_data=[]
    row_data=[]
    for elem in row_elems:
        text = elem.text.replace(",","")
        if(i%7!=0):
            if(text!="-"):
                text=float(text)
            else:
                text=None
            row_data.append(text)
            i+=1
        else:
            table_data.append(row_data)
            row_data=[]
            row_data.append(text)
            i+=1
    return table_data

def remove_multiple_spaces(string):
    if type(string)==str:
        return ' '.join(string.split())
    return string
def get_children(html_content):
    return [item for item in html_content.children if type(item)==bs4.element.Tag or len(str(item).replace("\n","").strip())>0]

def data():
    driver = webdriver.Chrome("C:/Users/HP/chromedriver.exe")
    url = "https://finance.yahoo.com/quote/%5ENSEI/history?p=%5ENSEI"
    driver.get(url)
    html = driver.page_source
    page_content = BeautifulSoup(html, "html.parser")
    table_rows=page_content.find_all('tr',attrs={'class':'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)'}) #.text.split(",")
    lis=[]
    table_string="<table><tr>"
    for row in table_rows:
        table_string+=str(row)
    table_string=table_string+"</tr></table>"
    head=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    content = BeautifulSoup(table_string,"html.parser")
    temp_dataset=get_table_simple(content)
    dataset=[]
    temp_dataset=temp_dataset
    dataset=temp_dataset
    return dataset

def dl():
    dataset = data()
    dataset.pop(0)
    dataset = pd.DataFrame(dataset, columns =["Date","Open","High","Low","Close","Adj Close","Volume"]) 
    dataset = dataset.fillna(method="bfill")
    print(dataset)
    datelist=dataset["Date"].tolist()
    openlist=dataset["Open"].tolist()
    highlist=dataset["High"].tolist()
    lowlist=dataset["Low"].tolist()
    adlist=dataset["Adj Close"].tolist()
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
    y=dataset["Close"]
    del dataset["Date"]
    del dataset["Close"]
    x=dataset
    y=y.to_frame()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1,shuffle = False, stratify = None)
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    val={
        "Predicted": y_pred.tolist(),
        "Actual": reduce(lambda z, y :z + y, y_test.values.tolist())
    }
    ans=pd.DataFrame(val)
    print("The mean absolute error is: ",mean_absolute_error(ans["Predicted"].values.tolist(), ans["Actual"].values.tolist()))
    return X_train, X_test, y_train, y_test,regressor,datelist,open2list,high2list,low2list,ad2list,vol2list



def formss(request):
    X_train, X_test, y_train, y_test,regressor,datelist,open2list,high2list,low2list,ad2list,vol2list=dl()
    if(request.method=='POST'):
        student=StudentForm(request.POST)
        if(student.is_valid()):
            open=student.cleaned_data.get("open")
            high=student.cleaned_data.get("high")
            low=student.cleaned_data.get("low")
            adjacent=student.cleaned_data.get("adjacent")
            vol=student.cleaned_data.get("vol")
            X_testt=[[open,high,low,adjacent,vol]]
            y_pred=regressor.predict(X_testt)
    else:
        student = StudentForm()
        y_pred=[0]
    passd={'form':student,
    'ans':y_pred,
    'date':datelist,'open':open2list,'high':high2list,'low':low2list,'adj':ad2list,'vol':vol2list}
    return render(request,"Stock/index.html/",passd)
def hello(request):
    return render(request,"Stock/index.html/")
def logo(request):
    return render(request,"Stock/logo.png/")
def js(request):
    return render(request,"Stock/cht.js/")
